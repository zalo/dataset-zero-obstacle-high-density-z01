# modal deploy scripts/deploy_api.py
#
# Serves the fine-tuned FLUX.2 Klein 4B model for PCB routing inference.
# Single web endpoint: / (test page), /route (POST), /status (GET), /samples (GET).
#
# The test page loads instantly. The model loads lazily on first generation
# request, streaming each loading step as SSE events so the page shows progress.

import io
import random
import threading

import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = modal.App("pcbrouter-flux2-short")

GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.34.2",
        "datasets~=3.2.0",
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
TEST_SAMPLE_INDICES = [0, 3, 6, 9, 12, 15]
HF_DATASET = "makeshifted/zero-obstacle-high-density-z01-training"

fastapi_app = FastAPI()


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
        """Minimal startup — no model loading. Page serves instantly."""
        self.pipe = None
        self.test_samples = None
        self.checkpoint_name = None
        self._model_lock = threading.Lock()
        self._samples_lock = threading.Lock()

    def _load_model(self, status_callback=None):
        """Lazy-load the model on first request. Thread-safe."""
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

            # Prefer the output root if a full pipeline was saved there (model_index.json).
            # Otherwise fall back to the latest checkpoint-N/ directory.
            model_index = os.path.join(FULL_OUTPUT_DIR, "model_index.json")
            if os.path.isfile(model_index):
                log("Found completed model at output root")
                log("Loading fine-tuned pipeline (this takes ~60s)...")
                self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                    FULL_OUTPUT_DIR,
                    torch_dtype=torch.bfloat16,
                )
                self.checkpoint_name = "final"
                log("Fine-tuned pipeline loaded")
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

                log("Loading base pipeline (this takes ~60s)...")
                self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                    FULL_MODEL_DIR,
                    torch_dtype=torch.bfloat16,
                )
                log("Base pipeline loaded")

                if checkpoint_path:
                    transformer_path = os.path.join(checkpoint_path, "transformer")
                    if os.path.isdir(transformer_path):
                        log(f"Loading fine-tuned transformer from {self.checkpoint_name}...")
                        from diffusers import Flux2Transformer2DModel

                        self.pipe.transformer = Flux2Transformer2DModel.from_pretrained(
                            transformer_path,
                            torch_dtype=torch.bfloat16,
                        )
                        log("Fine-tuned transformer loaded")
                    else:
                        log(f"Warning: no transformer dir in {self.checkpoint_name}")

            log("Moving model to GPU...")
            self.pipe.to("cuda")
            log("Model ready!")

    def _load_samples(self):
        """Lazy-load test samples from HuggingFace."""
        if self.test_samples is not None:
            return

        with self._samples_lock:
            if self.test_samples is not None:
                return

            import base64

            from datasets import load_dataset

            print("Loading test samples from HuggingFace...")
            ds = load_dataset(HF_DATASET, split="test")
            self.test_samples = []
            for idx in TEST_SAMPLE_INDICES:
                if idx < len(ds):
                    row = ds[idx]
                    buf_c = io.BytesIO()
                    row["cond_image"].save(buf_c, format="PNG")
                    buf_o = io.BytesIO()
                    row["output_image"].save(buf_o, format="PNG")
                    self.test_samples.append(
                        {
                            "idx": idx,
                            "cond_b64": base64.b64encode(buf_c.getvalue()).decode(),
                            "gt_b64": base64.b64encode(buf_o.getvalue()).decode(),
                        }
                    )
            print(f"Loaded {len(self.test_samples)} test samples")

    def _sse_generate(self, prompt, input_image, seed):
        """Stream model loading (if needed) + generation progress as SSE."""
        import base64
        import json
        import queue

        import torch

        q = queue.Queue()

        def status_callback(msg):
            q.put({"stage": "loading", "message": msg})

        # Load model in a thread so we can stream status
        def load_and_generate():
            try:
                self._load_model(status_callback=status_callback)

                seed_val = seed if seed is not None else random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed_val)

                def callback_on_step_end(pipe_self, step, timestep, callback_kwargs):
                    q.put(
                        {
                            "stage": "generating",
                            "step": step + 1,
                            "total": NUM_STEPS,
                            "progress": round((step + 1) / NUM_STEPS, 3),
                        }
                    )
                    return callback_kwargs

                q.put({"stage": "loading", "message": "Starting generation..."})

                images = self.pipe(
                    prompt=prompt,
                    image=input_image,
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
            return JSONResponse(
                {
                    "checkpoint": inference.checkpoint_name or "not loaded",
                    "model_loaded": inference.pipe is not None,
                    "samples_loaded": inference.test_samples is not None,
                }
            )

        @fastapi_app.get("/samples")
        async def samples():
            """Return test samples (lazy-loaded from HuggingFace on first call)."""
            inference._load_samples()
            return JSONResponse(inference.test_samples)

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

        @fastapi_app.get("/")
        async def test_page():
            return HTMLResponse(content=TEST_PAGE_HTML)

        return fastapi_app


TEST_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCB Router - Short Model (71 samples)</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.4em; color: #fff; }
  .subtitle { text-align: center; margin-bottom: 24px; color: #888; font-size: 0.9em; }
  .controls { display: flex; gap: 12px; justify-content: center; align-items: center; margin-bottom: 24px; flex-wrap: wrap; }
  .controls label { font-size: 0.85em; color: #aaa; }
  .controls input { background: #1a1a1a; border: 1px solid #333; color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 0.85em; }
  button { background: #2563eb; color: #fff; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9em; font-weight: 600; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 20px; }
  .card { background: #141414; border: 1px solid #222; border-radius: 8px; padding: 16px; }
  .card-header { font-size: 0.8em; color: #888; margin-bottom: 10px; }
  .images { display: flex; gap: 8px; align-items: flex-start; }
  .img-col { text-align: center; }
  .img-col img { width: 200px; height: 200px; border-radius: 4px; background: #000; image-rendering: pixelated; }
  .img-col .label { font-size: 0.7em; color: #666; margin-top: 4px; }
  .progress { height: 3px; background: #222; border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .progress-fill { height: 100%; background: #2563eb; width: 0%; transition: width 0.3s; }
  .status { font-size: 0.75em; color: #666; margin-top: 4px; text-align: right; min-height: 1.2em; }
  .loading-banner { text-align: center; padding: 40px; color: #888; font-size: 0.9em; }
  .loading-banner .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #333; border-top-color: #2563eb; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>PCB Trace Router (Short Model - 71 samples)</h1>
<p class="subtitle">Checkpoint: <strong id="checkpoint">loading...</strong></p>

<div class="controls">
  <label>Seed: <input type="number" id="seed" value="42" style="width:70px"></label>
  <button id="run-btn" onclick="runAll()" disabled>Loading samples...</button>
</div>

<div id="grid" class="grid">
  <div class="loading-banner"><span class="spinner"></span> Loading test samples from HuggingFace...</div>
</div>

<script>
let SAMPLES = [];
const ROUTE_URL = window.location.origin + '/route';

async function loadSamples() {
  try {
    const resp = await fetch('/samples');
    SAMPLES = await resp.json();
    renderGrid();
    document.getElementById('run-btn').disabled = false;
    document.getElementById('run-btn').textContent = 'Generate All';
  } catch (e) {
    document.getElementById('grid').innerHTML = '<div class="loading-banner">Failed to load samples: ' + e.message + '</div>';
  }
}

async function loadStatus() {
  try {
    const resp = await fetch('/status');
    const data = await resp.json();
    document.getElementById('checkpoint').textContent = data.checkpoint;
  } catch (e) {}
}

function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  SAMPLES.forEach((s, i) => {
    grid.innerHTML += `
      <div class="card" id="card-${i}">
        <div class="card-header">Test sample #${s.idx}</div>
        <div class="images">
          <div class="img-col">
            <img src="data:image/png;base64,${s.cond_b64}" />
            <div class="label">Input</div>
          </div>
          <div class="img-col">
            <img id="output-${i}" src="data:image/png;base64,${s.gt_b64}" style="opacity:0.3" />
            <div class="label" id="output-label-${i}">Ground Truth (faded)</div>
          </div>
          <div class="img-col">
            <img id="gt-${i}" src="data:image/png;base64,${s.gt_b64}" />
            <div class="label">Ground Truth</div>
          </div>
        </div>
        <div class="progress"><div class="progress-fill" id="prog-${i}"></div></div>
        <div class="status" id="status-${i}">Ready</div>
      </div>`;
  });
}

async function generate(idx) {
  const s = SAMPLES[idx];
  const seed = parseInt(document.getElementById('seed').value);
  const outImg = document.getElementById('output-' + idx);
  const outLabel = document.getElementById('output-label-' + idx);
  const prog = document.getElementById('prog-' + idx);
  const status = document.getElementById('status-' + idx);

  outImg.style.opacity = '0.3';
  outLabel.textContent = 'Generating...';
  prog.style.width = '0%';
  status.textContent = 'Connecting...';

  try {
    const resp = await fetch(ROUTE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ input_image: s.cond_b64, seed }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.stage === 'loading') {
            status.textContent = event.message;
          } else if (event.stage === 'generating') {
            const pct = Math.round(event.progress * 100);
            prog.style.width = pct + '%';
            status.textContent = `Step ${event.step}/${event.total}`;
          } else if (event.stage === 'complete') {
            outImg.src = 'data:image/png;base64,' + event.image;
            outImg.style.opacity = '1';
            outLabel.textContent = 'Model Output';
            prog.style.width = '100%';
            status.textContent = 'Done';
            if (event.checkpoint) {
              document.getElementById('checkpoint').textContent = event.checkpoint;
            }
          } else if (event.stage === 'error') {
            status.textContent = 'Error: ' + event.message;
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
}

async function runAll() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  for (let i = 0; i < SAMPLES.length; i++) {
    await generate(i);
  }

  btn.disabled = false;
  btn.textContent = 'Generate All';
}

// Load page content immediately, samples async
loadSamples();
loadStatus();
</script>
</body>
</html>"""
