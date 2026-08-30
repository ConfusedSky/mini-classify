/**
 * Dev-only render hook for the mini-classify renderer pilot (2026-08-29):
 * `window.__pilot.render(libPath, axis, states, variants)` parses a model once
 * and returns one base64 PNG per state per variant through the thumbnail
 * chain. Installed from main.tsx under `import.meta.env.DEV` only; nothing in
 * the app calls it.
 */
import type { CameraState, OrbitAxis } from '../../../shared/types'
import { disposeModel, formatOf, parseModel } from '../three/models'
import { renderThumbnail, type ThumbOptions } from '../three/renderer'
import { setLightingMode } from '../viewer/lighting'

async function blobToBase64(blob: Blob): Promise<string> {
  const buf = new Uint8Array(await blob.arrayBuffer())
  let s = ''
  for (let i = 0; i < buf.length; i += 0x8000) {
    s += String.fromCharCode(...buf.subarray(i, i + 0x8000))
  }
  return btoa(s)
}

export function installPilot(): void {
  setLightingMode('camera')
  const w = window as unknown as { __pilot?: unknown }
  w.__pilot = {
    async render(
      libPath: string,
      axis: OrbitAxis,
      states: CameraState[],
      variants: Record<string, ThumbOptions>,
    ): Promise<Record<string, string[]>> {
      const format = formatOf(libPath)
      if (format === null) throw new Error(`unsupported: ${libPath}`)
      const res = await fetch(`/api/file?path=${encodeURIComponent(libPath)}`)
      if (!res.ok) throw new Error(`fetch ${res.status}: ${libPath}`)
      const object = parseModel(await res.arrayBuffer(), format)
      const out: Record<string, string[]> = {}
      for (const [name, opts] of Object.entries(variants)) {
        out[name] = []
        for (const state of states) {
          out[name].push(await blobToBase64(await renderThumbnail(object, state, axis, opts)))
        }
      }
      disposeModel(object)
      return out
    },
  }
}
