# modal deploy scripts/deploy_api_short.py
#
# API-only inference endpoint for the fine-tuned FLUX.2 Klein 4B model.
# Routes: POST /route, GET /status
# The test UI is served from GitHub Pages (see docs/index.html).

import io
import random
import threading

import modal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = modal.App("pcbrouter-flux2-short")

GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.34.2",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]>=0.34.0",
        "peft>=0.17.0",
        "pillow>=10.0.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.51.0",
    )
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

FULL_MODEL_DIR = "/model-full"
FULL_OUTPUT_DIR = "/model-full/output"
NUM_STEPS = 50
DEFAULT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.cls(
    image=modal_image,
    gpu="L4",
    timeout=10 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        FULL_MODEL_DIR: modal.Volume.from_name("pcbrouter-flux2-klein-short-volume"),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)
class Inference:
    @modal.enter()
    def initialize(self):
        self.pipe = None
        self.checkpoint_name = None
        self._model_lock = threading.Lock()

    def _load_model(self, status_callback=None):
        if self.pipe is not None:
            return
        with self._model_lock:
            if self.pipe is not None:
                return

            import glob
            import os

            import diffusers
            import torch

            def log(msg):
                print(msg)
                if status_callback:
                    status_callback(msg)

            log("Finding best model weights...")
            model_index = os.path.join(FULL_OUTPUT_DIR, "model_index.json")
            if os.path.isfile(model_index):
                log("Found completed model at output root")
                log("Loading fine-tuned pipeline...")
                self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                    FULL_OUTPUT_DIR, torch_dtype=torch.bfloat16,
                )
                self.checkpoint_name = "final"
            else:
                checkpoints = glob.glob(os.path.join(FULL_OUTPUT_DIR, "checkpoint-*"))
                if checkpoints:
                    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))
                    checkpoint_path = checkpoints[-1]
                    self.checkpoint_name = os.path.basename(checkpoint_path)
                    log(f"Found {len(checkpoints)} checkpoints, using {self.checkpoint_name}")
                else:
                    checkpoint_path = None
                    self.checkpoint_name = "base"
                    log("No checkpoints found, using base model")

                log("Loading base pipeline...")
                self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                    FULL_MODEL_DIR, torch_dtype=torch.bfloat16,
                )

                if checkpoint_path:
                    transformer_path = os.path.join(checkpoint_path, "transformer")
                    if os.path.isdir(transformer_path):
                        log(f"Loading fine-tuned transformer from {self.checkpoint_name}...")
                        from diffusers import Flux2Transformer2DModel
                        self.pipe.transformer = Flux2Transformer2DModel.from_pretrained(
                            transformer_path, torch_dtype=torch.bfloat16,
                        )

            log("Moving model to GPU...")
            self.pipe.to("cuda")
            log("Model ready!")

    def _sse_generate(self, prompt, input_image, seed):
        import base64
        import json
        import queue

        import torch

        q = queue.Queue()

        def status_callback(msg):
            q.put({"stage": "loading", "message": msg})

        def load_and_generate():
            try:
                self._load_model(status_callback=status_callback)
                seed_val = seed if seed is not None else random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed_val)

                def callback_on_step_end(pipe_self, step, timestep, callback_kwargs):
                    q.put({
                        "stage": "generating",
                        "step": step + 1,
                        "total": NUM_STEPS,
                        "progress": round((step + 1) / NUM_STEPS, 3),
                    })
                    return callback_kwargs

                q.put({"stage": "loading", "message": "Starting generation..."})
                images = self.pipe(
                    prompt=prompt, image=input_image,
                    num_inference_steps=NUM_STEPS, guidance_scale=4.0,
                    height=256, width=256, generator=generator,
                    callback_on_step_end=callback_on_step_end,
                ).images

                with io.BytesIO() as buf:
                    images[0].save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                q.put({"stage": "complete", "image": img_b64, "checkpoint": self.checkpoint_name})
            except Exception as e:
                q.put({"stage": "error", "message": str(e)})
            finally:
                q.put(None)

        t = threading.Thread(target=load_and_generate)
        t.start()
        while True:
            event = q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        t.join()
        torch.cuda.empty_cache()

    @modal.asgi_app()
    def serve(self):
        from PIL import Image
        inference = self

        @fastapi_app.get("/status")
        async def status():
            return JSONResponse({
                "checkpoint": inference.checkpoint_name or "not loaded",
                "model_loaded": inference.pipe is not None,
            })

        @fastapi_app.post("/route")
        async def route(request: Request):
            import base64
            body = await request.json()
            if "input_image" not in body:
                return JSONResponse(status_code=400, content={"error": "input_image is required"})

            input_b64 = body["input_image"]
            instruction = body.get("instruction", DEFAULT_INSTRUCTION)
            seed = body.get("seed")

            input_bytes = base64.b64decode(input_b64)
            input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
            input_image = input_image.resize((256, 256))

            return StreamingResponse(
                inference._sse_generate(instruction, input_image, seed),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return fastapi_app
