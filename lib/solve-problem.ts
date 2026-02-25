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
import {
  CELL_SIZE_MM,
  IMAGE_SIZE_PX,
  TRACE_MARGIN_MM,
  TRACE_THICKNESS_MM,
  VIA_DIAMETER_MM,
} from "./generator-params.ts"

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

export type SolveProblemImageOptions = Omit<SolveProblemOptions, "outputDir">

export type SolveProblemImageArtifacts = {
  boundary_connection_pairs: BoundaryConnectionPair[]
  routed_paths: HighDensityIntraNodeRoute[]
  connection_pair_svg: string
  routed_svg: string
  connection_pair_png: Uint8Array
  routed_png: Uint8Array
}

export type SolveProblemImageResult =
  | {
      ok: true
      artifacts: SolveProblemImageArtifacts
    }
  | {
      ok: false
      reason: string
      error?: unknown
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
  const solved = await solveProblemToImageArtifacts(problem, options)
  if (!solved.ok) {
    return solved
  }

  const imagesDir = join(options.outputDir, "images")
  const connectionPairImagePath = join(
    imagesDir,
    "connection-pairs",
    `${problem.problemId}.png`,
  )
  const routedImagePath = join(imagesDir, "routed", `${problem.problemId}.png`)

  await mkdir(dirname(connectionPairImagePath), { recursive: true })
  await writeFile(
    connectionPairImagePath.replace(/\.png$/, ".svg"),
    solved.artifacts.connection_pair_svg,
    "utf8",
  )
  await writeFile(connectionPairImagePath, solved.artifacts.connection_pair_png)
  await mkdir(dirname(routedImagePath), { recursive: true })
  await writeFile(
    routedImagePath.replace(/\.png$/, ".svg"),
    solved.artifacts.routed_svg,
    "utf8",
  )
  await writeFile(routedImagePath, solved.artifacts.routed_png)

  return {
    ok: true,
    row: {
      boundary_connection_pairs: solved.artifacts.boundary_connection_pairs,
      connection_pair_image: toPosixRelativePath(
        options.outputDir,
        connectionPairImagePath,
      ),
      routed_image: toPosixRelativePath(options.outputDir, routedImagePath),
      routed_paths: solved.artifacts.routed_paths,
    },
  }
}

export async function solveProblemToImageArtifacts(
  problem: GeneratedProblem,
  options: SolveProblemImageOptions,
): Promise<SolveProblemImageResult> {
  const imageSizePx = options.imageSizePx ?? IMAGE_SIZE_PX
  const viaDiameterMm = options.viaDiameterMm ?? VIA_DIAMETER_MM
  const solver = new HighDensitySolverA01({
    nodeWithPortPoints: problem.nodeWithPortPoints,
    cellSizeMm: options.cellSizeMm ?? CELL_SIZE_MM,
    viaDiameter: viaDiameterMm,
    traceThickness: options.traceThicknessMm ?? TRACE_THICKNESS_MM,
    traceMargin: options.traceMarginMm ?? TRACE_MARGIN_MM,
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
    cellSizeMm: options.cellSizeMm ?? CELL_SIZE_MM,
    viaDiameter: viaDiameterMm,
    traceThickness: options.traceThicknessMm ?? TRACE_THICKNESS_MM,
    traceMargin: options.traceMarginMm ?? TRACE_MARGIN_MM,
  }) as RuntimeSolver
  pairedSolver.setup()

  const connectionPairPointRadiusPx = mmToImagePx({
    lengthMm: viaDiameterMm / 2,
    imageSizePx,
    boardWidthMm: problem.nodeWithPortPoints.width,
    boardHeightMm: problem.nodeWithPortPoints.height,
  })
  const connectionPairStrokeWidthPx = mmToImagePx({
    lengthMm: viaDiameterMm / 2,
    imageSizePx,
    boardWidthMm: problem.nodeWithPortPoints.width,
    boardHeightMm: problem.nodeWithPortPoints.height,
  })
  const netStrokeColorOffset =
    hashSeed(`${problem.problemId}:net-color-offset`) %
    Math.max(problem.boundaryConnectionPairs.length, 1)

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

  const connectionPairImage = renderGraphicsImage({
    graphics: inputGraphics,
    center: problem.nodeWithPortPoints.center,
    width: problem.nodeWithPortPoints.width,
    height: problem.nodeWithPortPoints.height,
    sizePx: imageSizePx,
    pointRadiusPx: connectionPairPointRadiusPx,
    netStrokeColorOffset,
    netStrokeWidthPx: connectionPairStrokeWidthPx,
  })

  const routedImage = renderGraphicsImage({
    graphics: routedGraphics,
    center: problem.nodeWithPortPoints.center,
    width: problem.nodeWithPortPoints.width,
    height: problem.nodeWithPortPoints.height,
    sizePx: imageSizePx,
    pointRadiusPx: connectionPairPointRadiusPx,
    netStrokeColorOffset,
    netStrokeWidthPx: connectionPairStrokeWidthPx,
  })

  return {
    ok: true,
    artifacts: {
      boundary_connection_pairs: problem.boundaryConnectionPairs,
      routed_paths: routedPaths,
      connection_pair_svg: connectionPairImage.svg,
      routed_svg: routedImage.svg,
      connection_pair_png: connectionPairImage.pngBytes,
      routed_png: routedImage.pngBytes,
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
  netStrokeColorOffset?: number
  netStrokeWidthPx?: number
}): Promise<void> {
  const rendered = renderGraphicsImage(params)
  const svgPath = params.pngFilePath.replace(/\.png$/, ".svg")
  await mkdir(dirname(params.pngFilePath), { recursive: true })
  await writeFile(svgPath, rendered.svg, "utf8")
  await writeFile(params.pngFilePath, rendered.pngBytes)
}

function renderGraphicsImage(params: {
  graphics: GraphicsObject
  center: { x: number; y: number }
  width: number
  height: number
  sizePx: number
  pointRadiusPx?: number
  netStrokeColorOffset?: number
  netStrokeWidthPx?: number
}): { svg: string; pngBytes: Uint8Array } {
  const {
    graphics,
    center,
    width,
    height,
    sizePx,
    pointRadiusPx,
    netStrokeColorOffset,
    netStrokeWidthPx,
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

  if (netStrokeColorOffset !== undefined) {
    rawSvg = setPointCircleNetStroke(rawSvg, {
      connectionIndexOffset: netStrokeColorOffset,
      strokeWidthPx: netStrokeWidthPx ?? Math.max(1, sizePx * 0.003),
    })
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

  const resvg = new Resvg(croppedSvg, {
    fitTo: {
      mode: "width",
      value: sizePx,
    },
  })
  const pngBytes = new Uint8Array(resvg.render().asPng())

  return {
    svg: croppedSvg,
    pngBytes,
  }
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

  const withViewport = svg
    .replace(/viewBox="[^"]*"/, `viewBox="${viewBox}"`)
    .replace(/width="[^"]*"/, `width="${outputWidth}"`)
    .replace(/height="[^"]*"/, `height="${outputHeight}"`)

  return withViewport.replace(
    /<rect\s+width="100%"\s+height="100%"\s+fill="white"\s*\/>/,
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="white"/>`,
  )
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

function setPointCircleNetStroke(
  svg: string,
  params: {
    connectionIndexOffset: number
    strokeWidthPx: number
  },
): string {
  const { connectionIndexOffset, strokeWidthPx } = params
  const strokeWidthValue = Number(strokeWidthPx.toFixed(3)).toString()

  return svg.replace(/<circle\b[^>]*>/g, (circleTag) => {
    if (!circleTag.includes('data-type="point"')) {
      return circleTag
    }

    const labelMatch = /\bdata-label="([^"]+)"/.exec(circleTag)
    const label = labelMatch?.[1] ?? ""
    const connectionIndex = getConnectionIndex(label)
    if (connectionIndex === undefined) {
      return circleTag
    }
    const netColor = getNetColor(connectionIndex + connectionIndexOffset)

    let updatedTag = setSvgAttribute(circleTag, "stroke", netColor)
    updatedTag = setSvgAttribute(updatedTag, "stroke-width", strokeWidthValue)
    return updatedTag
  })
}

function getConnectionIndex(connectionLabel: string): number | undefined {
  const suffixMatch = /_(\d+)$/.exec(connectionLabel)
  if (suffixMatch) {
    return Number.parseInt(suffixMatch[1] ?? "0", 10)
  }

  return undefined
}

function getNetColor(netIndex: number): string {
  const normalizedIndex = Number.isFinite(netIndex) ? netIndex : 0
  const hue = (((normalizedIndex * 137.508) % 360) + 360) % 360
  return hslToHex(hue, 0.85, 0.4)
}

function setSvgAttribute(tag: string, attr: string, value: string): string {
  const attrPattern = new RegExp(`\\b${attr}="[^"]*"`)
  if (attrPattern.test(tag)) {
    return tag.replace(attrPattern, `${attr}="${value}"`)
  }

  return tag.replace(/\/>$/, ` ${attr}="${value}"/>`)
}

function hslToHex(
  hueDeg: number,
  saturation: number,
  lightness: number,
): string {
  const h = (((hueDeg % 360) + 360) % 360) / 360
  const s = Math.max(0, Math.min(1, saturation))
  const l = Math.max(0, Math.min(1, lightness))

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q

  const r = hueToRgb(p, q, h + 1 / 3)
  const g = hueToRgb(p, q, h)
  const b = hueToRgb(p, q, h - 1 / 3)

  return `#${toHexChannel(r)}${toHexChannel(g)}${toHexChannel(b)}`
}

function hueToRgb(p: number, q: number, tInput: number): number {
  let t = tInput
  if (t < 0) t += 1
  if (t > 1) t -= 1
  if (t < 1 / 6) return p + (q - p) * 6 * t
  if (t < 1 / 2) return q
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
  return p
}

function toHexChannel(value: number): string {
  const channel = Math.round(Math.max(0, Math.min(1, value)) * 255)
  return channel.toString(16).padStart(2, "0")
}
