"""The ModernGL counterpart to renderer_open3d.py: can we pick the GPU, own the
buffers, and hold more than one context?

Filament cannot reach the 4060 because its GL backend asks for a default EGL
display; NVIDIA headless wants eglQueryDevicesEXT + EGL_PLATFORM_DEVICE_EXT.
ModernGL asks that way, so device_index selects the card directly. This also
checks the thing that actually blocks the actor design — a second renderer in
one process, which core-dumps under Open3D (LEARNINGS.md, commit 6683399).

Needs its own venv; moderngl and trimesh are deliberately not project deps:
  uv venv /tmp/mgl && uv pip install --python /tmp/mgl/bin/python moderngl trimesh
  /tmp/mgl/bin/python eval/renderer_moderngl.py [device_index] [mesh.stl]

The shading here is a hand-written Lambert pass, NOT Filament's defaultLit with
IBL, so the frame time is not directly comparable to renderer_open3d.py. Upload
and residency numbers are. See docs/actor-refactor/renderer_alternatives.md.
"""
import subprocess
import sys
import time

import numpy as np
import moderngl
import trimesh

DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MESH = sys.argv[2] if len(sys.argv) > 2 else "test-stls/bunny.stl"
SUBDIV = 3
SIZE = 512

VERT = """#version 330
uniform mat4 mvp; in vec3 in_v; in vec3 in_n; out vec3 n;
void main() { n = in_n; gl_Position = mvp * vec4(in_v, 1.0); }"""

FRAG = """#version 330
in vec3 n; out vec4 f; uniform vec3 sun;
void main() { float d = max(dot(normalize(n), sun), 0.0);
              f = vec4(vec3(0.7) * (0.15 + 0.85 * d), 1.0); }"""


def vram():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(q.stdout.strip().splitlines()[0]) if q.returncode == 0 else -1


# 1. Device selection. Walk the indices to see what this machine exposes.
for i in range(4):
    try:
        c = moderngl.create_context(standalone=True, backend="egl", device_index=i)
        print(f"device_index={i}: {c.info['GL_RENDERER']}")
        c.release()
    except Exception as e:
        print(f"device_index={i}: unavailable ({type(e).__name__})")

ctx = moderngl.create_context(standalone=True, backend="egl", device_index=DEVICE)
print(f"\nusing device_index={DEVICE}: {ctx.info['GL_RENDERER']}")

m = trimesh.load(MESH)
for _ in range(SUBDIV):
    m = m.subdivide()
verts = np.asarray(m.vertices, dtype="f4")
faces = np.asarray(m.faces, dtype="i4")
# numpy vertex normals: trimesh's no-scipy fallback is unusably slow at this size
fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
              verts[faces[:, 2]] - verts[faces[:, 0]])
norms = np.zeros_like(verts)
for k in range(3):
    np.add.at(norms, faces[:, k], fn)
norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-20)
data = np.hstack([verts, norms]).astype("f4")
per_copy_mb = (data.nbytes + faces.nbytes) / 1e6
print(f"{len(faces):,} tris, {len(verts):,} verts, {per_copy_mb:.0f} MB per copy")

# 2. Explicit upload, and a frame.
t = time.perf_counter()
vbo, ibo = ctx.buffer(data.tobytes()), ctx.buffer(faces.tobytes())
ctx.finish()
print(f"{(time.perf_counter() - t) * 1000:9.1f} ms  upload")

prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_v", "in_n")], ibo)
fbo = ctx.framebuffer(color_attachments=[ctx.texture((SIZE, SIZE), 4)],
                      depth_attachment=ctx.depth_texture((SIZE, SIZE)))
fbo.use()
ctx.enable(moderngl.DEPTH_TEST)
prog["mvp"].write(np.eye(4, dtype="f4").tobytes())
prog["sun"].value = (0.0, 0.0, 1.0)

fbo.clear(1, 1, 1, 1)
vao.render()
ctx.finish()
t = time.perf_counter()
for _ in range(20):
    fbo.clear(1, 1, 1, 1)
    vao.render()
ctx.finish()
print(f"{(time.perf_counter() - t) / 20 * 1000:9.2f} ms  per {SIZE}x{SIZE} frame")

# 3. Residency we can see and evict on demand.
print("\n--- residency ---")
base = vram()
keep = []
for i in range(8):
    keep.append((ctx.buffer(data.tobytes()), ctx.buffer(faces.tobytes())))
    ctx.finish()
    print(f"  {i + 1} resident: {vram()} MiB (+{vram() - base}, "
          f"expected +{per_copy_mb * (i + 1):.0f})")
for b in [x for pair in keep[:4] for x in pair]:
    b.release()
ctx.finish()
print(f"  after releasing 4: {vram()} MiB (+{vram() - base})")

# 4. The row that matters: a second context, which Open3D cannot survive.
print("\n--- second context in the same process ---")
try:
    ctx2 = moderngl.create_context(standalone=True, backend="egl", device_index=DEVICE)
    f2 = ctx2.framebuffer(color_attachments=[ctx2.texture((256, 256), 4)])
    f2.use()
    f2.clear(1, 0, 0, 1)
    ctx2.finish()
    print(f"  ok, both alive: {ctx2.info['GL_RENDERER']}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print("exited without abort")
