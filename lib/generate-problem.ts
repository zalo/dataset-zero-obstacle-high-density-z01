import type { NodeWithPortPoints, PortPoint } from "@tscircuit/high-density-a01"

type BoundarySide = "left" | "right" | "top" | "bottom"

export type BoundaryEndpoint = PortPoint & {
  side: BoundarySide
}

export type BoundaryConnectionPair = {
  connectionName: string
  start: BoundaryEndpoint
  end: BoundaryEndpoint
}

export type GeneratedProblem = {
  problemId: string
  nodeWithPortPoints: NodeWithPortPoints
  boundaryConnectionPairs: BoundaryConnectionPair[]
}

export type GenerateProblemOptions = {
  problemId?: string
  widthMm?: number
  heightMm?: number
  pairCount?: number
  minPairCount?: number
  maxPairCount?: number
  availableZ?: number[]
  seed?: number
}

export function generateProblem(
  options: GenerateProblemOptions = {},
): GeneratedProblem {
  const widthMm = options.widthMm ?? 10
  const heightMm = options.heightMm ?? 10
  const minPairCount = options.minPairCount ?? 2
  const maxPairCount = options.maxPairCount ?? 10
  const availableZ = options.availableZ ?? [0, 1]
  const pairCount =
    options.pairCount ??
    randomInt(createRng(options.seed ?? Date.now()), minPairCount, maxPairCount)

  const rng = createRng(options.seed ?? Date.now())
  const problemId = options.problemId ?? `p-${Date.now().toString(36)}`
  const halfWidth = widthMm / 2
  const halfHeight = heightMm / 2

  const usedBoundarySlots = new Set<string>()
  const boundaryConnectionPairs: BoundaryConnectionPair[] = []

  for (let i = 0; i < pairCount; i += 1) {
    const connectionName = `conn_${i.toString().padStart(3, "0")}`

    const startSide = randomSide(rng)
    const start = createUniqueBoundaryEndpoint({
      rng,
      halfWidth,
      halfHeight,
      usedBoundarySlots,
      availableZ,
      connectionName,
      side: startSide,
    })

    const endSide = rng() < 0.7 ? oppositeSide(startSide) : randomSide(rng)
    const end = createUniqueBoundaryEndpoint({
      rng,
      halfWidth,
      halfHeight,
      usedBoundarySlots,
      availableZ,
      connectionName,
      side: endSide,
    })

    boundaryConnectionPairs.push({
      connectionName,
      start,
      end,
    })
  }

  const portPoints = boundaryConnectionPairs.flatMap((pair) => [
    pair.start,
    pair.end,
  ])

  return {
    problemId,
    boundaryConnectionPairs,
    nodeWithPortPoints: {
      capacityMeshNodeId: `node-${problemId}`,
      center: { x: 0, y: 0 },
      width: widthMm,
      height: heightMm,
      availableZ,
      portPoints,
    },
  }
}

function createUniqueBoundaryEndpoint(params: {
  rng: () => number
  halfWidth: number
  halfHeight: number
  usedBoundarySlots: Set<string>
  availableZ: number[]
  connectionName: string
  side: BoundarySide
}): BoundaryEndpoint {
  const {
    rng,
    halfWidth,
    halfHeight,
    usedBoundarySlots,
    availableZ,
    connectionName,
  } = params

  let side = params.side

  for (let attempt = 0; attempt < 200; attempt += 1) {
    const endpoint = createBoundaryEndpoint({
      rng,
      halfWidth,
      halfHeight,
      connectionName,
      side,
      z: availableZ[randomInt(rng, 0, availableZ.length - 1)] ?? 0,
    })

    const boundarySlotKey = getBoundarySlotKey(endpoint)
    if (!usedBoundarySlots.has(boundarySlotKey)) {
      usedBoundarySlots.add(boundarySlotKey)
      return endpoint
    }

    side = randomSide(rng)
  }

  throw new Error("Unable to generate unique boundary endpoint after 200 tries")
}

function createBoundaryEndpoint(params: {
  rng: () => number
  halfWidth: number
  halfHeight: number
  connectionName: string
  side: BoundarySide
  z: number
}): BoundaryEndpoint {
  const { rng, halfWidth, halfHeight, connectionName, side, z } = params

  if (side === "left" || side === "right") {
    return {
      side,
      connectionName,
      x: side === "left" ? -halfWidth : halfWidth,
      y: roundToGrid(randomFloat(rng, -halfHeight, halfHeight), 0.02),
      z,
    }
  }

  return {
    side,
    connectionName,
    x: roundToGrid(randomFloat(rng, -halfWidth, halfWidth), 0.02),
    y: side === "bottom" ? -halfHeight : halfHeight,
    z,
  }
}

function getBoundarySlotKey(endpoint: BoundaryEndpoint): string {
  return `${endpoint.side}:${endpoint.x.toFixed(3)}:${endpoint.y.toFixed(3)}:${endpoint.z}`
}

function roundToGrid(value: number, step: number): number {
  return Math.round(value / step) * step
}

function randomFloat(rng: () => number, min: number, max: number): number {
  return min + (max - min) * rng()
}

function randomInt(rng: () => number, min: number, max: number): number {
  return Math.floor(randomFloat(rng, min, max + 1))
}

function randomSide(rng: () => number): BoundarySide {
  const sides: BoundarySide[] = ["left", "right", "top", "bottom"]
  return sides[Math.floor(rng() * sides.length)] ?? "left"
}

function oppositeSide(side: BoundarySide): BoundarySide {
  switch (side) {
    case "left":
      return "right"
    case "right":
      return "left"
    case "top":
      return "bottom"
    case "bottom":
      return "top"
  }
}

function createRng(seed: number): () => number {
  let state = seed >>> 0
  if (state === 0) {
    state = 0x6d2b79f5
  }

  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000
  }
}
