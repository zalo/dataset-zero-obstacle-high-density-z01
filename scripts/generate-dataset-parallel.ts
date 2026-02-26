import { existsSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { parseArgs } from "node:util"
import {
  isMainThread,
  parentPort,
  Worker,
  workerData,
} from "node:worker_threads"

import {
  type GeneratedProblem,
  generateProblem,
} from "../lib/generate-problem.ts"
import {
  CELL_SIZE_MM,
  IMAGE_SIZE_PX,
  MAX_CONNECTIONS,
  MAX_SOLVE_ATTEMPTS,
  MIN_CONNECTIONS,
  TRACE_MARGIN_MM,
  TRACE_THICKNESS_MM,
  VIA_DIAMETER_MM,
} from "../lib/generator-params.ts"
import { type DatasetRow, solveProblem } from "../lib/solve-problem.ts"

type WorkerRange = {
  startIndex: number
  endIndex: number
}

type WorkerPayload = {
  range: WorkerRange
  outputDir: string
}

type WorkerResult = {
  rows: DatasetRowWithId[]
  failures: Array<{ problemId: string; reason: string }>
  skippedIds: string[]
  attempts: number
}

type WorkerMessage =
  | { type: "progress"; skipped: boolean }
  | ({ type: "done" } & WorkerResult)

type DatasetRowWithId = DatasetRow & { id: string }

if (isMainThread) {
  await runMainThread()
} else {
  await runWorkerThread()
}

async function runMainThread(): Promise<void> {
  const { values: args } = parseArgs({
    options: {
      "sample-count": { type: "string" },
      "output-dir": { type: "string" },
      concurrency: { type: "string" },
      offset: { type: "string" },
    },
    strict: true,
  })

  if (!args["sample-count"]) {
    throw new Error("Missing required flag: --sample-count")
  }
  if (!args["output-dir"]) {
    throw new Error("Missing required flag: --output-dir")
  }
  if (!args.concurrency) {
    throw new Error("Missing required flag: --concurrency")
  }

  const sampleCount = parsePositiveInt(args["sample-count"])
  const outputDir = args["output-dir"]
  const concurrency = parsePositiveInt(args.concurrency)
  const offset = args.offset ? parseNonNegativeInt(args.offset) : 0
  const effectiveConcurrency = Math.min(sampleCount, concurrency)

  const imagesDir = join(outputDir, "images")
  await mkdir(join(imagesDir, "connection-pairs"), { recursive: true })
  await mkdir(join(imagesDir, "routed"), { recursive: true })

  const existingRows = await loadExistingRows(join(outputDir, "dataset.jsonl"))

  const startedAt = Date.now()
  let completedSamples = 0
  let skippedSamples = 0
  const onProgress = (skipped: boolean) => {
    completedSamples += 1
    if (skipped) skippedSamples += 1
    const generated = completedSamples - skippedSamples
    const rate = samplesPerMinute(generated, startedAt).toFixed(0)
    process.stdout.write(
      `\r${completedSamples}/${sampleCount} samples (${rate} samples/minute)`,
    )
  }

  const workerRanges = buildWorkerRanges(sampleCount, effectiveConcurrency, offset)
  const workerRuns = workerRanges.map((range) =>
    runWorker({ range, outputDir }, onProgress),
  )
  const workerResults = await Promise.all(workerRuns)
  process.stdout.write("\n")

  const skippedIds = workerResults.flatMap((result) => result.skippedIds)
  const skippedRows = skippedIds
    .map((id) => existingRows.get(id))
    .filter((row): row is DatasetRowWithId => row != null)
  const newRows = workerResults.flatMap((result) => result.rows)
  const rows = [...skippedRows, ...newRows]
  const failures = workerResults.flatMap((result) => result.failures)
  const attempts = workerResults.reduce(
    (sum, result) => sum + result.attempts,
    0,
  )

  rows.sort((a, b) => a.id.localeCompare(b.id))

  const jsonlPath = join(outputDir, "dataset.jsonl")
  await writeFile(
    jsonlPath,
    `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  )

  const metadata = {
    created_at: new Date().toISOString(),
    requested_samples: sampleCount,
    generated_samples: rows.length,
    attempts,
    skipped: failures.length,
    concurrency,
    workers_used: effectiveConcurrency,
    elapsed_ms: Date.now() - startedAt,
    samples_per_minute: samplesPerMinute(newRows.length, startedAt),
    columns: [
      "boundary_connection_pairs",
      "connection_pair_image",
      "routed_image",
      "routed_paths",
    ],
  }

  await writeFile(
    join(outputDir, "metadata.json"),
    `${JSON.stringify(metadata, null, 2)}\n`,
    "utf8",
  )

  if (failures.length > 0) {
    await writeFile(
      join(outputDir, "failures.json"),
      `${JSON.stringify(failures, null, 2)}\n`,
      "utf8",
    )
  }

  if (rows.length < sampleCount) {
    console.warn(`generated ${rows.length}/${sampleCount} samples`)
  }

  console.log(`dataset written to ${jsonlPath}`)
  console.log(
    `samples/minute: ${samplesPerMinute(newRows.length, startedAt).toFixed(2)}`,
  )
}

async function runWorkerThread(): Promise<void> {
  const payload = workerData as WorkerPayload
  const rows: DatasetRowWithId[] = []
  const failures: Array<{ problemId: string; reason: string }> = []
  const skippedIds: string[] = []
  let attempts = 0

  for (
    let index = payload.range.startIndex;
    index <= payload.range.endIndex;
    index += 1
  ) {
    const sampleId = `sample-${index.toString().padStart(6, "0")}`

    const routedPngPath = join(payload.outputDir, "images", "routed", `${sampleId}.png`)
    if (existsSync(routedPngPath)) {
      skippedIds.push(sampleId)
      parentPort?.postMessage({ type: "progress", skipped: true } satisfies WorkerMessage)
      continue
    }

    const pairCount = pairCountForIndex(index)

    let solved = false
    let lastReason = "Failed to generate/solve sample"

    for (let attempt = 1; attempt <= MAX_SOLVE_ATTEMPTS; attempt += 1) {
      attempts += 1

      const seed = 1109 + index * 1000 + attempt
      let problem: GeneratedProblem
      try {
        problem = generateProblem({
          problemId: sampleId,
          seed,
          pairCount,
          minPointSeparationMm: VIA_DIAMETER_MM,
        })
      } catch (error) {
        lastReason =
          error instanceof Error ? error.message : "Failed to generate problem"
        continue
      }

      const solvedResult = await solveProblem(problem, {
        outputDir: payload.outputDir,
        imageSizePx: IMAGE_SIZE_PX,
        cellSizeMm: CELL_SIZE_MM,
        viaDiameterMm: VIA_DIAMETER_MM,
        traceThicknessMm: TRACE_THICKNESS_MM,
        traceMarginMm: TRACE_MARGIN_MM,
      })

      if (solvedResult.ok) {
        rows.push({
          id: sampleId,
          ...solvedResult.row,
        })
        solved = true
        break
      }

      lastReason = solvedResult.reason
    }

    if (!solved) {
      failures.push({
        problemId: sampleId,
        reason: lastReason,
      })
    }

    parentPort?.postMessage({ type: "progress", skipped: false } satisfies WorkerMessage)
  }

  parentPort?.postMessage({
    type: "done",
    rows,
    failures,
    skippedIds,
    attempts,
  } satisfies WorkerMessage)
}

function runWorker(
  payload: WorkerPayload,
  onProgress: (skipped: boolean) => void,
): Promise<WorkerResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), { workerData: payload })

    let settled = false
    worker.on("message", (msg: WorkerMessage) => {
      if (msg.type === "progress") {
        onProgress(msg.skipped)
        return
      }
      settled = true
      resolve(msg)
    })
    worker.on("error", reject)
    worker.on("exit", (code) => {
      if (!settled && code !== 0) {
        reject(new Error(`Worker stopped with exit code ${code}`))
      }
    })
  })
}

function buildWorkerRanges(
  sampleCount: number,
  concurrency: number,
  offset: number,
): WorkerRange[] {
  const workerCount = Math.max(1, Math.min(sampleCount, concurrency))
  const baseSize = Math.floor(sampleCount / workerCount)
  const remainder = sampleCount % workerCount

  const ranges: WorkerRange[] = []
  let cursor = offset + 1
  for (let i = 0; i < workerCount; i += 1) {
    const size = baseSize + (i < remainder ? 1 : 0)
    const startIndex = cursor
    const endIndex = cursor + size - 1
    ranges.push({ startIndex, endIndex })
    cursor = endIndex + 1
  }

  return ranges
}

function parsePositiveInt(raw: string): number {
  const value = Number(raw)
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Expected positive integer, got: ${raw}`)
  }

  return Math.floor(value)
}

function parseNonNegativeInt(raw: string): number {
  const value = Number(raw)
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Expected non-negative integer, got: ${raw}`)
  }

  return Math.floor(value)
}

function pairCountForIndex(index: number): number {
  const hash = ((index * 1103515245 + 12345) >>> 16) & 0x7fff
  const range = MAX_CONNECTIONS - MIN_CONNECTIONS + 1
  return MIN_CONNECTIONS + (hash % range)
}

async function loadExistingRows(
  path: string,
): Promise<Map<string, DatasetRowWithId>> {
  const map = new Map<string, DatasetRowWithId>()
  if (!existsSync(path)) return map
  const content = await readFile(path, "utf8")
  for (const line of content.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      const row = JSON.parse(trimmed) as DatasetRowWithId
      if (row.id) map.set(row.id, row)
    } catch {}
  }
  return map
}

function samplesPerMinute(completedCount: number, startedAtMs: number): number {
  const elapsedMinutes = Math.max((Date.now() - startedAtMs) / 60000, 1 / 60000)
  return completedCount / elapsedMinutes
}
