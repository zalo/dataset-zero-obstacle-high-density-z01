# modal run scripts/train_flux2_full.py
#
# FLUX.2 Klein 4B FULL fine-tuning for PCB routing image-to-image editing.
#
# Uses the dedicated Klein img2img training script from diffusers which
# properly handles paired (input_image, output_image, instruction) triplets.
# The script is patched to do full fine-tuning instead of LoRA.
#
# At inference time the model is used in img2img mode: pass a connection-pairs
# image and the edit instruction, get back a routed image.
#
# Adapted from morphmaker.ai/morphmaker_train_flux2_full.py
#
# Training tips incorporated from PixelWave FLUX.1-dev community results:
#   - Low learning rate (1e-6 range) to avoid fuzzy/washed outputs
#   - constant_with_warmup scheduler for stability
#   - FP8 training for memory efficiency
#   - Gradient checkpointing

from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App(name="pcbrouter-train-flux2-klein-full")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.34.2",
    "datasets~=3.2.0",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "huggingface-hub>=0.34.0",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft>=0.17.0",
    "pillow>=10.0.0",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "transformers>=4.51.0",
    "torch~=2.5.0",
    "torchvision~=0.20",
    "triton~=3.1.0",
    "wandb==0.17.6",
)

# Diffusers SHA: "Support Flux Klein peft (fal) lora format" (2026-02-21).
# Includes Klein img2img training script + fal LoRA format fix.
# Intentionally before the transformers v5 migration (2026-02-24) for stability.
GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

image = (
    image.apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
    # Add patch scripts and apply them to the Klein img2img training script
    .add_local_file(
        Path(__file__).parent / "patch_klein_full_ft.py",
        remote_path="/root/patch_klein_full_ft.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "patch_disable_precache.py",
        remote_path="/root/patch_disable_precache.py",
        copy=True,
    )
    .run_commands(
        # Convert the img2img LoRA script to full fine-tuning
        "python3 /root/patch_klein_full_ft.py "
        "/root/examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py "
        "/root/examples/dreambooth/train_dreambooth_flux2_klein_img2img_full.py",
        # Disable pre-caching to avoid OOM
        "python3 /root/patch_disable_precache.py "
        "/root/examples/dreambooth/train_dreambooth_flux2_klein_img2img_full.py",
    )
)


@dataclass
class SharedConfig:
    """Configuration shared across project components."""

    model_name: str = "black-forest-labs/FLUX.2-klein-base-4B"


volume = modal.Volume.from_name(
    "pcbrouter-flux2-klein-full-volume", create_if_missing=True
)
MODEL_DIR = "/model"
OUTPUT_DIR = "/model/output"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

USE_WANDB = True


@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,
)
def download_models(config):
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
    )
    volume.commit()


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for full fine-tuning on PCB routing paired image data.

    Hyperparameters informed by community results on Flux fine-tuning:
    - Very low LR (1e-6) to prevent fuzzy/washed outputs with full FT
    - constant_with_warmup scheduler for stable training on small datasets
    - FP8 training to reduce VRAM usage
    - Gradient checkpointing for memory efficiency
    """

    # HuggingFace dataset with paired images (cond_image, output_image,
    # instruction columns).
    hf_training_dataset: str = "tscircuit/zero-obstacle-high-density-z01"

    resolution: int = 256
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # effective batch size = 4
    learning_rate: float = 1e-5  # same range as morphmaker full FT
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    # 18195 train images / effective_batch 4 = ~4549 steps/epoch, * 3 epochs ≈ 13647
    max_train_steps: int = 13650
    checkpointing_steps: int = 2000
    seed: int = 42


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume},
    timeout=21600,  # 6 hours
    secrets=[huggingface_secret]
    + (
        [
            modal.Secret.from_name(
                "wandb-secret", required_keys=["WANDB_API_KEY"]
            )
        ]
        if USE_WANDB
        else []
    ),
)
def train(config, segment_end: int, resume: bool):
    import subprocess

    from accelerate.utils import write_basic_config

    write_basic_config(mixed_precision="bf16")

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        exitcode = process.wait()
        if exitcode != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print(f"launching FLUX.2 Klein img2img FULL fine-tuning (steps up to {segment_end})")
    cmd = [
        "accelerate",
        "launch",
        "examples/dreambooth/train_dreambooth_flux2_klein_img2img_full.py",
        "--mixed_precision=bf16",
        f"--pretrained_model_name_or_path={MODEL_DIR}",
        f"--dataset_name={config.hf_training_dataset}",
        f"--output_dir={OUTPUT_DIR}",
        "--image_column=output_image",
        "--cond_image_column=cond_image",
        "--caption_column=instruction",
        "--instance_prompt=Route the traces between the color matched pins",
        f"--resolution={config.resolution}",
        f"--train_batch_size={config.train_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        "--gradient_checkpointing",
        f"--learning_rate={config.learning_rate}",
        f"--lr_scheduler={config.lr_scheduler}",
        f"--lr_warmup_steps={config.lr_warmup_steps}",
        f"--max_train_steps={segment_end}",
        f"--checkpointing_steps={config.checkpointing_steps}",
        f"--seed={config.seed}",
    ]
    if resume:
        cmd.append("--resume_from_checkpoint=latest")
    if USE_WANDB:
        cmd.append("--report_to=wandb")

    _exec_subprocess(cmd)
    volume.commit()


@app.local_entrypoint()
def run(
    max_train_steps: int = 13650,
):
    import subprocess

    config = TrainConfig(max_train_steps=max_train_steps)

    print("downloading FLUX.2 Klein base model")
    download_models.remote(SharedConfig())

    # Train in segments of checkpointing_steps, deploying the inference
    # API after each checkpoint so the model can be tested during training.
    step = 0
    while step < max_train_steps:
        segment_end = min(step + config.checkpointing_steps, max_train_steps)
        resume = step > 0

        print(f"\n{'='*60}")
        print(f"Training segment: steps {step} -> {segment_end}")
        print(f"{'='*60}")
        train.remote(config, segment_end=segment_end, resume=resume)

        print(f"\nCheckpoint at step {segment_end} committed. Deploying inference API...")
        subprocess.run(
            ["modal", "deploy", "scripts/deploy_api.py"],
            check=True,
        )
        print(f"Inference API deployed with checkpoint at step {segment_end}.")

        step = segment_end

    print("\nTraining finished. Final model deployed.")
