# modal deploy scripts/deploy_api.py
#
# Serves the fine-tuned FLUX.2 Klein 4B model for PCB routing inference.
# Takes an input connection-pairs image and produces a routed image using
# img2img mode.  Progress is streamed via SSE.
#
# Adapted from morphmaker.ai/morphmaker_api_flux2.py

import io
import random

import modal

app = modal.App("pcbrouter-flux2")

# Diffusers SHA: "Support Flux Klein peft (fal) lora format" (2026-02-21).
# Before the transformers v5 migration (2026-02-24) for stability.
GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

image = (
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
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",
        }
    )
)

with image.imports():
    import base64
    import json
    import queue
    import threading

    import diffusers
    import torch
    from fastapi import Request
    from fastapi.responses import StreamingResponse
    from PIL import Image

FULL_MODEL_DIR = "/model-full"
FULL_OUTPUT_DIR = "/model-full/output"

# Klein base models are undistilled — use ~50 steps with CFG 4.0 for quality.
NUM_STEPS = 50

DEFAULT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)


def _sse_generate(pipe, prompt, input_image, strength, seed):
    """Run img2img pipeline in a thread, yield SSE events with progress."""
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    q = queue.Queue()

    # Effective steps for img2img = NUM_STEPS * strength
    effective_steps = max(1, int(NUM_STEPS * strength))

    def callback_on_step_end(pipe_self, step, timestep, callback_kwargs):
        progress = (step + 1) / effective_steps
        q.put(
            {
                "stage": "generating",
                "step": step + 1,
                "total": effective_steps,
                "progress": round(progress, 3),
            }
        )
        return callback_kwargs

    def run_pipeline():
        try:
            images = pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=NUM_STEPS,
                guidance_scale=4.0,
                height=256,
                width=256,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            ).images
            with io.BytesIO() as buf:
                images[0].save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            q.put({"stage": "complete", "image": img_b64})
        except Exception as e:
            q.put({"stage": "error", "message": str(e)})
        finally:
            q.put(None)

    t = threading.Thread(target=run_pipeline)
    t.start()

    while True:
        event = q.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"

    t.join()
    torch.cuda.empty_cache()


@app.cls(
    image=image,
    gpu="L4",
    timeout=5 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        FULL_MODEL_DIR: modal.Volume.from_name("pcbrouter-flux2-klein-full-volume"),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)
class Inference:
    @modal.enter()
    def initialize(self):
        import glob
        import os

        # Load from the latest checkpoint, or the final output if training is complete.
        # Checkpoints are saved as OUTPUT_DIR/checkpoint-{step}/
        checkpoints = sorted(glob.glob(os.path.join(FULL_OUTPUT_DIR, "checkpoint-*")))
        model_path = checkpoints[-1] if checkpoints else FULL_OUTPUT_DIR
        print(f"Loading model from: {model_path}")

        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def route(self, request: Request):
        """Accept a connection-pairs image and return a routed image.

        POST body (JSON):
            input_image: base64-encoded PNG of the connection-pairs image
            instruction: (optional) edit instruction text
            strength:    (optional) img2img strength, 0.0-1.0 (default 0.75)
            seed:        (optional) RNG seed for reproducibility
        """
        body = await request.json()

        if "input_image" not in body:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=400,
                content={"error": "input_image is required"},
            )

        input_b64 = body["input_image"]
        instruction = body.get("instruction", DEFAULT_INSTRUCTION)
        strength = min(max(float(body.get("strength", 0.75)), 0.0), 1.0)
        seed = body.get("seed")

        # Decode input image and resize to match training resolution
        input_bytes = base64.b64decode(input_b64)
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        input_image = input_image.resize((256, 256))

        print(
            f"Inference: instruction={instruction!r}, "
            f"strength={strength}, seed={seed}, "
            f"image_size={input_image.size}"
        )

        return StreamingResponse(
            _sse_generate(self.pipe, instruction, input_image, strength, seed),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
