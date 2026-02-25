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
