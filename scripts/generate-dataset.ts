import { appendFile, mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"

import { generateProblem } from "../lib/generate-problem.ts"
import { solveProblem } from "../lib/solve-problem.ts"

const sampleCount = parseSampleCount(process.argv[2])
const outputDir = process.argv[3] ?? "dataset"
const failures: Array<{ problemId: string; reason: string }> = []
const pairCountBySampleIndex = new Map<number, number>()

const MIN_CONNECTIONS = 2
const MAX_CONNECTIONS = 10

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
  const problem = generateProblem({
    problemId,
    seed: 1109 + attempts,
    pairCount,
  })

  const result = await solveProblem(problem, {
    outputDir,
    imageSizePx: 1024,
    cellSizeMm: 0.1,
    viaDiameterMm: 0.3,
    traceThicknessMm: 0.12,
    traceMarginMm: 0.06,
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

function parseSampleCount(rawArg: string | undefined): number {
  const parsed = Number(rawArg ?? "100")
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(
      `Invalid sample count: ${rawArg ?? "<empty>"}. Usage: bun scripts/generate-dataset.ts <count> [outputDir]`,
    )
  }

  return Math.floor(parsed)
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
