import { appendFile, mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { parseArgs } from "node:util"

import {
  type GeneratedProblem,
  generateProblem,
} from "../lib/generate-problem.ts"
import {
  CELL_SIZE_MM,
  VIA_DIAMETER_MM,
  TRACE_THICKNESS_MM,
  TRACE_MARGIN_MM,
  IMAGE_SIZE_PX,
  MIN_CONNECTIONS,
  MAX_CONNECTIONS,
} from "../lib/generator-params.ts"
import { solveProblem } from "../lib/solve-problem.ts"

const { values: args } = parseArgs({
  options: {
    "sample-count": { type: "string" },
    "output-dir": { type: "string" },
  },
  strict: true,
})

if (!args["sample-count"]) {
  throw new Error("Missing required flag: --sample-count")
}
if (!args["output-dir"]) {
  throw new Error("Missing required flag: --output-dir")
}

const sampleCount = parsePositiveInt(args["sample-count"])
const outputDir = args["output-dir"]
const failures: Array<{ problemId: string; reason: string }> = []
const pairCountBySampleIndex = new Map<number, number>()

await mkdir(outputDir, { recursive: true })
await mkdir(join(outputDir, "images", "connection-pairs"), { recursive: true })
await mkdir(join(outputDir, "images", "routed"), { recursive: true })

const jsonlPath = join(outputDir, "dataset.jsonl")
await writeFile(jsonlPath, "", "utf8")

let solvedCount = 0
let attempts = 0
const maxAttempts = sampleCount * 200

while (solvedCount < sampleCount && attempts < maxAttempts) {
  attempts += 1

  const sampleIndex = solvedCount + 1
  const pairCount = getOrCreatePairCount(sampleIndex)
  const problemId = `sample-${sampleIndex.toString().padStart(6, "0")}`
  let problem: GeneratedProblem
  try {
    problem = generateProblem({
      problemId,
      seed: 1109 + attempts,
      pairCount,
      minPointSeparationMm: VIA_DIAMETER_MM,
    })
  } catch (error) {
    const reason =
      error instanceof Error ? error.message : "Failed to generate problem"
    failures.push({
      problemId,
      reason,
    })
    console.warn(`retry ${problemId}: ${reason}`)
    continue
  }

  const result = await solveProblem(problem, {
    outputDir,
    imageSizePx: IMAGE_SIZE_PX,
    cellSizeMm: CELL_SIZE_MM,
    viaDiameterMm: VIA_DIAMETER_MM,
    traceThicknessMm: TRACE_THICKNESS_MM,
    traceMarginMm: TRACE_MARGIN_MM,
  })

  if (!result.ok) {
    failures.push({
      problemId,
      reason: result.reason,
    })

    console.warn(`retry ${problemId}: ${result.reason}`)
    continue
  }

  solvedCount += 1
  const row = {
    id: problemId,
    ...result.row,
  }

  await appendFile(jsonlPath, `${JSON.stringify(row)}\n`, "utf8")
  console.log(`ok ${problemId} (${solvedCount}/${sampleCount})`)
}

const metadata = {
  created_at: new Date().toISOString(),
  requested_samples: sampleCount,
  generated_samples: solvedCount,
  attempts,
  skipped: failures.length,
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

if (solvedCount < sampleCount) {
  console.warn(
    `generated ${solvedCount}/${sampleCount} before hitting max attempts (${maxAttempts})`,
  )
}

console.log(`dataset written to ${jsonlPath}`)

function parsePositiveInt(raw: string): number {
  const value = Number(raw)
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Expected positive integer, got: ${raw}`)
  }

  return Math.floor(value)
}

function getOrCreatePairCount(sampleIndex: number): number {
  const existing = pairCountBySampleIndex.get(sampleIndex)
  if (existing !== undefined) {
    return existing
  }

  const created = randomInt(MIN_CONNECTIONS, MAX_CONNECTIONS)
  pairCountBySampleIndex.set(sampleIndex, created)
  return created
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min
}
