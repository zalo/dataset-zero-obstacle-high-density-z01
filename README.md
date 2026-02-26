# dataset-zero-obstacle-10mm-square-z01

Synthetic dataset generator for high-density routing problems on a 10x10mm square.

Each row written to `dataset/dataset.jsonl` has these main columns:

- `boundary_connection_pairs`
- `connection_pair_image`
- `routed_image`
- `routed_paths`

## Repo Structure

- `lib/generate-problem.ts`: creates one random 10x10mm boundary-pair routing problem.
- `lib/solve-problem.ts`: runs `HighDensitySolverA01`, skips failures, and renders SVG+PNG outputs via `graphics-debug` + `resvg`.
- `scripts/generate-dataset.ts`: generates `N` samples, writes JSONL + metadata + image files.

## Install

```bash
bun install
```

## Generate Dataset

```bash
bun run generate:dataset -- 100
```

Optional output directory:

```bash
bun run generate:dataset -- 100 ./dataset-hd-v1
```

## Output Layout

```text
dataset/
  dataset.jsonl
  metadata.json
  failures.json
  images/
    connection-pairs/
      sample-000001.png
      sample-000001.svg
    routed/
      sample-000001.png
      sample-000001.svg
```

## Notes

- The solver can fail for some generated problems; those are skipped and logged.
- PNG images are rendered from the solver `visualize()` graphics object.
- SVG viewBox is cropped to the exact 10x10mm board region before rasterization.

## Cloudflare Endpoint

`cloudflare-endpoint/` contains a Worker endpoint that:

- accepts `POST /generate` requests,
- runs the sample generator+solver,
- returns dataset row data plus SVG payloads for both image types,
- caches responses in Cloudflare KV using a SHA-256 hash of normalized input.

### Deploy

```bash
bun run cloudflare:deploy
```

### Generate Against Worker

```bash
bun run generate:against-server -- 1000 https://<your-worker>.workers.dev ./dataset-against-server 80
```

All four CLI args are required: sample count, endpoint URL, output dir, concurrency.

This script writes the same dataset layout as `generate-dataset.ts`, including:

- `dataset.jsonl` + `metadata.json` (+ `failures.json` when needed),
- `images/connection-pairs/*.svg` + `*.png`,
- `images/routed/*.svg` + `*.png`.
