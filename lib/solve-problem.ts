import { mkdir, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Resvg } from "@resvg/resvg-js"
import {
  type HighDensityIntraNodeRoute,
  HighDensitySolverA01,
} from "@tscircuit/high-density-a01"
import {
  type GraphicsObject,
  getBounds,
  getSvgFromGraphicsObject,
} from "graphics-debug"

import type {
  BoundaryConnectionPair,
  GeneratedProblem,
} from "./generate-problem.ts"

const GRAPHICS_PADDING_PX = 40

export type DatasetRow = {
  boundary_connection_pairs: BoundaryConnectionPair[]
  connection_pair_image: string
  routed_image: string
  routed_paths: HighDensityIntraNodeRoute[]
}

export type SolveProblemOptions = {
  outputDir: string
  imageSizePx?: number
  cellSizeMm?: number
  viaDiameterMm?: number
  traceThicknessMm?: number
  traceMarginMm?: number
}

export type SolveProblemResult =
  | {
      ok: true
      row: DatasetRow
    }
  | {
      ok: false
      reason: string
      error?: unknown
    }

type RuntimeSolver = HighDensitySolverA01 & {
  solve: () => void
  setup: () => void
  solved: boolean
  failed: boolean
  error: unknown
}

export async function solveProblem(
  problem: GeneratedProblem,
  options: SolveProblemOptions,
): Promise<SolveProblemResult> {
  const imageSizePx = options.imageSizePx ?? 1024
  const viaDiameterMm = options.viaDiameterMm ?? 0.3
  const solver = new HighDensitySolverA01({
    nodeWithPortPoints: problem.nodeWithPortPoints,
    cellSizeMm: options.cellSizeMm ?? 0.1,
    viaDiameter: viaDiameterMm,
    traceThickness: options.traceThicknessMm ?? 0.12,
    traceMargin: options.traceMarginMm ?? 0.05,
    hyperParameters: {
      shuffleSeed: hashSeed(problem.problemId),
    },
  }) as RuntimeSolver

  try {
    solver.solve()
  } catch (error) {
    return {
      ok: false,
      reason: "Solver threw while solving",
      error,
    }
  }

  if (solver.failed || !solver.solved) {
    const errorMessage =
      typeof solver.error === "string"
        ? solver.error
        : solver.error instanceof Error
          ? solver.error.message
          : null

    return {
      ok: false,
      reason: errorMessage ?? "Solver did not finish with a solution",
      error: solver.error,
    }
  }

  const routedPaths = solver.getOutput()
  if (routedPaths.length !== problem.boundaryConnectionPairs.length) {
    return {
      ok: false,
      reason: `Incomplete route set (${routedPaths.length}/${problem.boundaryConnectionPairs.length})`,
    }
  }

  const pairedSolver = new HighDensitySolverA01({
    nodeWithPortPoints: problem.nodeWithPortPoints,
    cellSizeMm: options.cellSizeMm ?? 0.1,
    viaDiameter: viaDiameterMm,
    traceThickness: options.traceThicknessMm ?? 0.12,
    traceMargin: options.traceMarginMm ?? 0.05,
  }) as RuntimeSolver
  pairedSolver.setup()

  const connectionPairPointRadiusPx = mmToImagePx({
    lengthMm: viaDiameterMm / 2,
    imageSizePx,
    boardWidthMm: problem.nodeWithPortPoints.width,
    boardHeightMm: problem.nodeWithPortPoints.height,
  })

  const inputGraphics = withBoardOutline(
    pairedSolver.visualize(),
    problem.nodeWithPortPoints.center,
    problem.nodeWithPortPoints.width,
    problem.nodeWithPortPoints.height,
  )

  const routedGraphics = withBoardOutline(
    solver.visualize(),
    problem.nodeWithPortPoints.center,
    problem.nodeWithPortPoints.width,
    problem.nodeWithPortPoints.height,
  )

  const imagesDir = join(options.outputDir, "images")
  const connectionPairImagePath = join(
    imagesDir,
    "connection-pairs",
    `${problem.problemId}.png`,
  )
  const routedImagePath = join(imagesDir, "routed", `${problem.problemId}.png`)

  await renderAndWriteGraphicsImage({
    graphics: inputGraphics,
    center: problem.nodeWithPortPoints.center,
    width: problem.nodeWithPortPoints.width,
    height: problem.nodeWithPortPoints.height,
    pngFilePath: connectionPairImagePath,
    sizePx: imageSizePx,
    pointRadiusPx: connectionPairPointRadiusPx,
  })

  await renderAndWriteGraphicsImage({
    graphics: routedGraphics,
    center: problem.nodeWithPortPoints.center,
    width: problem.nodeWithPortPoints.width,
    height: problem.nodeWithPortPoints.height,
    pngFilePath: routedImagePath,
    sizePx: imageSizePx,
  })

  return {
    ok: true,
    row: {
      boundary_connection_pairs: problem.boundaryConnectionPairs,
      connection_pair_image: toPosixRelativePath(
        options.outputDir,
        connectionPairImagePath,
      ),
      routed_image: toPosixRelativePath(options.outputDir, routedImagePath),
      routed_paths: routedPaths,
    },
  }
}

async function renderAndWriteGraphicsImage(params: {
  graphics: GraphicsObject
  center: { x: number; y: number }
  width: number
  height: number
  pngFilePath: string
  sizePx: number
  pointRadiusPx?: number
}): Promise<void> {
  const {
    graphics,
    center,
    width,
    height,
    pngFilePath,
    sizePx,
    pointRadiusPx,
  } = params

  let rawSvg = getSvgFromGraphicsObject(graphics, {
    backgroundColor: "white",
    includeTextLabels: false,
    svgWidth: sizePx,
    svgHeight: sizePx,
  })

  if (pointRadiusPx !== undefined) {
    rawSvg = setPointCircleRadius(rawSvg, pointRadiusPx)
  }

  const croppedSvg = cropSvgToBoardRegion({
    svg: rawSvg,
    graphics,
    center,
    width,
    height,
    svgWidth: sizePx,
    svgHeight: sizePx,
  })

  const svgPath = pngFilePath.replace(/\.png$/, ".svg")
  await mkdir(dirname(pngFilePath), { recursive: true })
  await writeFile(svgPath, croppedSvg, "utf8")

  const resvg = new Resvg(croppedSvg, {
    fitTo: {
      mode: "width",
      value: sizePx,
    },
  })
  const pngBytes = resvg.render().asPng()
  await writeFile(pngFilePath, pngBytes)
}

function cropSvgToBoardRegion(params: {
  svg: string
  graphics: GraphicsObject
  center: { x: number; y: number }
  width: number
  height: number
  svgWidth: number
  svgHeight: number
}): string {
  const { svg, graphics, center, width, height, svgWidth, svgHeight } = params

  const bounds = getBounds(graphics)
  const matrix = createProjectionMatrix({
    bounds,
    coordinateSystem: graphics.coordinateSystem ?? "cartesian",
    svgWidth,
    svgHeight,
    padding: GRAPHICS_PADDING_PX,
  })

  const halfWidth = width / 2
  const halfHeight = height / 2

  const boardCorners = [
    { x: center.x - halfWidth, y: center.y - halfHeight },
    { x: center.x + halfWidth, y: center.y - halfHeight },
    { x: center.x - halfWidth, y: center.y + halfHeight },
    { x: center.x + halfWidth, y: center.y + halfHeight },
  ]

  const projected = boardCorners.map((point) => projectPoint(matrix, point))
  const minX = Math.min(...projected.map((point) => point.x))
  const maxX = Math.max(...projected.map((point) => point.x))
  const minY = Math.min(...projected.map((point) => point.y))
  const maxY = Math.max(...projected.map((point) => point.y))

  return rewriteSvgViewport(svg, {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
    outputWidth: svgWidth,
    outputHeight: svgHeight,
  })
}

function rewriteSvgViewport(
  svg: string,
  params: {
    x: number
    y: number
    width: number
    height: number
    outputWidth: number
    outputHeight: number
  },
): string {
  const { x, y, width, height, outputWidth, outputHeight } = params
  const viewBox = `${x} ${y} ${width} ${height}`

  return svg
  // return svg
  //   .replace(/viewBox="[^"]*"/, `viewBox="${viewBox}"`)
  //   .replace(/width="[^"]*"/, `width="${outputWidth}"`)
  //   .replace(/height="[^"]*"/, `height="${outputHeight}"`)
}

function createProjectionMatrix(params: {
  bounds: { minX: number; maxX: number; minY: number; maxY: number }
  coordinateSystem: "cartesian" | "screen"
  svgWidth: number
  svgHeight: number
  padding: number
}): {
  scaleX: number
  scaleY: number
  translateX: number
  translateY: number
} {
  const { bounds, coordinateSystem, svgWidth, svgHeight, padding } = params
  const width = bounds.maxX - bounds.minX || 1
  const height = bounds.maxY - bounds.minY || 1
  const scale = Math.min(
    (svgWidth - 2 * padding) / width,
    (svgHeight - 2 * padding) / height,
  )

  const centerX = bounds.minX + width / 2
  const centerY = bounds.minY + height / 2

  return {
    scaleX: scale,
    scaleY: coordinateSystem === "screen" ? scale : -scale,
    translateX: svgWidth / 2 - scale * centerX,
    translateY:
      coordinateSystem === "screen"
        ? svgHeight / 2 - scale * centerY
        : svgHeight / 2 + scale * centerY,
  }
}

function projectPoint(
  matrix: {
    scaleX: number
    scaleY: number
    translateX: number
    translateY: number
  },
  point: { x: number; y: number },
): { x: number; y: number } {
  return {
    x: matrix.scaleX * point.x + matrix.translateX,
    y: matrix.scaleY * point.y + matrix.translateY,
  }
}

function withBoardOutline(
  graphics: GraphicsObject,
  center: { x: number; y: number },
  width: number,
  height: number,
): GraphicsObject {
  return {
    ...graphics,
    rects: [
      ...(graphics.rects ?? []),
      {
        center,
        width,
        height,
        fill: "none",
        stroke: "rgba(15, 23, 42, 0.8)",
      },
    ],
  }
}

function toPosixRelativePath(baseDir: string, absolutePath: string): string {
  const withoutBase = absolutePath.startsWith(baseDir)
    ? absolutePath.slice(baseDir.length + 1)
    : absolutePath

  return withoutBase.replaceAll("\\", "/")
}

function hashSeed(value: string): number {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }

  return hash >>> 0
}

function mmToImagePx(params: {
  lengthMm: number
  imageSizePx: number
  boardWidthMm: number
  boardHeightMm: number
}): number {
  const { lengthMm, imageSizePx, boardWidthMm, boardHeightMm } = params
  const drawableWidthPx = Math.max(imageSizePx - GRAPHICS_PADDING_PX * 2, 1)
  const pxPerMm = Math.min(
    drawableWidthPx / boardWidthMm,
    drawableWidthPx / boardHeightMm,
  )
  return lengthMm * pxPerMm
}

function setPointCircleRadius(svg: string, radiusPx: number): string {
  const radiusValue = Number(radiusPx.toFixed(3)).toString()

  return svg.replace(/<circle\b[^>]*>/g, (circleTag) => {
    if (!circleTag.includes('data-type="point"')) {
      return circleTag
    }

    return circleTag.replace(/\br="[^"]*"/, `r="${radiusValue}"`)
  })
}
