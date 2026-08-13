"""Which renderer calls hold the GIL, and for how long?

`renderer_moderngl.py` settles that ModernGL holds several contexts per process
where Open3D core-dumps on the second. That removes the *crash*, but contexts
are only half of what a threaded Renderer needs: if the draw and readback calls
hold the GIL, N contexts in one process still serialize on Python bytecode and
the actor design needs processes anyway.

Method: a background thread counts in pure Python. A call that releases the GIL
lets the counter keep climbing; one that holds it stalls the counter for its
duration. Reported as a fraction of a baseline taken while the main thread sleeps
(`time.sleep` releases the GIL, so that is the ceiling).

Two backends, because they live in different venvs:

  .venv/bin/python eval/renderer_gil.py open3d [size]
  /tmp/mgl/bin/python eval/renderer_gil.py moderngl [device_index] [size]

  # the moderngl venv, same as renderer_moderngl.py:
  uv venv /tmp/mgl && uv pip install --python /tmp/mgl/bin/python moderngl trimesh

Open3D's `render_to_image` is one call covering render *and* readback, so it is
also split against `np.asarray` / `Image.fromarray` — the line it shares in
`classify_stls.py:139`. ModernGL splits naturally into draw / finish / read.

Only the full per-view sequence is trustworthy. Measured in isolation and
repeated back-to-back, the readback calls give internally inconsistent numbers
(ModernGL's `fbo.read` came out slower at 512px than at 2048px), because a
readback with no draw behind it does not hit the same path. The combined row is
what the pipeline actually pays. Single-process, 8 iterations, one machine — run
it a few times, the spread is real.

See docs/masa/renderer_alternatives.md.
"""
import sys
import threading
import time

import numpy as np

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "open3d"

counter = 0
stop = False
BASE = 1.0


def spinner():
    global counter
    while not stop:
        counter += 1


def measure(label, fn, n=8):
    """Wall clock per call, and what fraction of the GIL the caller left free."""
    global counter
    c0, t0 = counter, time.perf_counter()
    for _ in range(n):
        out = fn()
    wall = time.perf_counter() - t0
    free = (counter - c0) / wall / BASE
    held = max(0.0, 1.0 - free)
    print(f"{label:28} {wall / n * 1000:7.1f} ms/call   GIL held {held * 100:3.0f}%"
          f"   ({wall / n * 1000 * held:5.1f} ms of GIL)")
    return out


def start_spinner():
    global BASE
    threading.Thread(target=spinner, daemon=True).start()
    time.sleep(0.3)
    c0, t0 = counter, time.perf_counter()
    time.sleep(0.5)                      # sleep releases the GIL: this is the ceiling
    BASE = (counter - c0) / (time.perf_counter() - t0)
    print(f"{'baseline (main asleep)':28} {'':>7}              {BASE / 1000:.0f}k spins/s\n")


def run_open3d(size):
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    from PIL import Image

    # exactly one OffscreenRenderer per process — a second aborts (LEARNINGS)
    r = rendering.OffscreenRenderer(size, size)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=200)
    mesh.compute_vertex_normals()
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    r.scene.add_geometry("mesh", mesh, mat)
    b = mesh.get_axis_aligned_bounding_box()
    c = b.get_center()
    r.setup_camera(45.0, c, c + [np.linalg.norm(b.get_extent()) * 1.4, 0, 0], [0, 0, 1])
    print(f"open3d  {len(mesh.triangles):,} tris  {size}px  (defaultLit + IBL)\n")

    start_spinner()
    img = measure("render_to_image", lambda: r.render_to_image())
    arr = measure("np.asarray", lambda: np.asarray(img))
    measure("Image.fromarray", lambda: Image.fromarray(arr))
    measure("full view (line 139)",
            lambda: Image.fromarray(np.asarray(r.render_to_image())))


VERT = """#version 330
uniform mat4 mvp; in vec3 in_v; in vec3 in_n; out vec3 n;
void main() { n = in_n; gl_Position = mvp * vec4(in_v, 1.0); }"""

FRAG = """#version 330
in vec3 n; out vec4 f; uniform vec3 sun;
void main() { float d = max(dot(normalize(n), sun), 0.0);
              f = vec4(vec3(0.7) * (0.15 + 0.85 * d), 1.0); }"""


def run_moderngl(device, size):
    import moderngl
    import trimesh

    ctx = moderngl.create_context(standalone=True, backend="egl", device_index=device)
    mesh = trimesh.creation.icosphere(subdivisions=6)
    verts = np.asarray(mesh.vertices, dtype="f4")
    norms = np.asarray(mesh.vertex_normals, dtype="f4")
    faces = np.asarray(mesh.faces, dtype="i4")
    print(f"moderngl  device_index={device}  {ctx.info['GL_RENDERER']}")
    print(f"          {len(faces):,} tris  {size}px  (hand-written Lambert)\n")

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    vbo = ctx.buffer(np.hstack([verts, norms]).astype("f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_v", "in_n")],
                           ctx.buffer(faces.tobytes()))
    prog["mvp"].write(np.eye(4, dtype="f4").tobytes())
    prog["sun"].value = (0.4, 0.4, 0.8)

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((size, size), 4)],
                          depth_attachment=ctx.depth_renderbuffer((size, size)))
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    def draw():
        fbo.clear(1, 1, 1, 1)
        vao.render()

    draw(); ctx.finish()                 # warm up: compile, allocate, first draw
    start_spinner()
    measure("vao.render (issue)", draw)
    measure("ctx.finish (gpu wait)", lambda: ctx.finish())
    measure("fbo.read (readback)", lambda: fbo.read())
    measure("full view (all three)", lambda: (draw(), ctx.finish(), fbo.read()))


if BACKEND == "open3d":
    run_open3d(int(sys.argv[2]) if len(sys.argv) > 2 else 2048)
elif BACKEND == "moderngl":
    run_moderngl(int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 2048)
else:
    sys.exit(f"unknown backend {BACKEND!r} — open3d or moderngl")

stop = True
