"""Generate report GIFs using the same Taichi GGUI view as main.py."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from fluid_sim import COLOR_SPEED, FlipFluid3D


OUT_DIR = Path(__file__).resolve().parent / "results"
FRAME_COUNT = 180
STEPS_PER_FRAME = cfg.SUBSTEPS
WATER_LIFT = 0.25
GIF_DURATION_MS = round(1000 / 60)
OUTPUT_SIZE = (640, 400)

SHAPE_NAMES = {
    cfg.INIT_SHAPE_CUBE: "cube",
    cfg.INIT_SHAPE_SPHERE: "sphere",
    cfg.INIT_SHAPE_HEART: "heart",
    cfg.INIT_SHAPE_BUNNY: "bunny",
}

COLOR_NAMES = {
    cfg.COLOR_SPEED: "speed",
    cfg.COLOR_DENSITY: "density",
    cfg.COLOR_PRESSURE: "pressure",
}


def init_taichi() -> None:
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)


def camera_pos(yaw: float, pitch: float, distance: float) -> tuple[float, float, float]:
    target = (0.5, 0.45, 0.5)
    cp = math.cos(pitch)
    return (
        target[0] + distance * math.sin(yaw) * cp,
        target[1] + distance * math.sin(pitch),
        target[2] + distance * math.cos(yaw) * cp,
    )


def lift_water(sim: FlipFluid3D) -> None:
    pos = sim.p_pos.to_numpy()
    pos[:, 1] += WATER_LIFT
    sim.p_pos.from_numpy(pos)


def prepare_case(shape: int, with_obstacle: bool) -> FlipFluid3D:
    sim = FlipFluid3D(shape)
    lift_water(sim)
    if with_obstacle:
        sim.set_obstacle(0.36, 0.32, 0.50, 0.0, 0.0, 0.0)
    else:
        sim.set_obstacle(0.90, 0.88, 0.90, 0.0, 0.0, 0.0)
    return sim


def render_main_view(
    window: ti.ui.Window,
    canvas: ti.ui.Canvas,
    scene: ti.ui.Scene,
    camera: ti.ui.Camera,
    sim: FlipFluid3D,
    shape: int,
    with_obstacle: bool,
) -> Image.Image:
    yaw = 0.78
    pitch = 0.34
    distance = 1.75
    dt = cfg.DEFAULT_DT
    flip_ratio = cfg.DEFAULT_FLIP_RATIO
    color_mode = COLOR_SPEED

    camera.position(*camera_pos(yaw, pitch, distance))
    camera.lookat(0.5, 0.45, 0.5)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(45)

    scene.set_camera(camera)
    scene.ambient_light((0.38, 0.42, 0.48))
    scene.point_light(pos=(0.35, 1.55, 1.1), color=(1.0, 1.0, 1.0))
    scene.lines(sim.box_edge_vertices, width=2.2, color=(0.78, 0.9, 1.0))
    scene.particles(sim.p_pos, radius=cfg.PARTICLE_RADIUS, per_vertex_color=sim.p_color)
    scene.particles(
        sim.obstacle_render_pos,
        radius=cfg.OBSTACLE_RADIUS,
        color=(0.95, 0.95, 0.88),
    )
    canvas.scene(scene)

    window.GUI.begin("3D FLIP Fluid", 0.015, 0.015, 0.32, 0.32)
    window.GUI.text(f"dt: {dt:.5f}  ([ / ])")
    window.GUI.text(f"flipRatio: {flip_ratio:.2f}  (, / . or 1/2/3)")
    window.GUI.text(f"shape: {SHAPE_NAMES[shape]}  (V)")
    window.GUI.text(f"color: {COLOR_NAMES[color_mode]}  (C)")
    window.GUI.text(f"avg divergence: {sim.avg_divergence_numpy():.5f}")
    window.GUI.text("RUNNING")
    window.GUI.text("obstacle demo" if with_obstacle else "normal falling demo")
    window.GUI.text("generated with main.py GGUI renderer")
    window.GUI.end()

    image = np.clip(window.get_image_buffer_as_numpy(), 0.0, 1.0)
    frame = Image.fromarray((image * 255).astype(np.uint8))
    frame = frame.transpose(Image.Transpose.ROTATE_90)
    frame = frame.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    window.show()
    return frame.convert("RGB")


def render_gif(
    window: ti.ui.Window,
    canvas: ti.ui.Canvas,
    scene: ti.ui.Scene,
    camera: ti.ui.Camera,
    shape: int,
    with_obstacle: bool,
) -> None:
    sim = prepare_case(shape, with_obstacle)
    suffix = "obstacle" if with_obstacle else "normal"
    frames: list[Image.Image] = []

    for frame_index in range(FRAME_COUNT):
        frame = render_main_view(window, canvas, scene, camera, sim, shape, with_obstacle)
        # Prevent GIF encoders/viewers from coalescing nearly identical frames.
        frame.putpixel((frame.width - 1, frame.height - 1), (frame_index % 256, 0, 0))
        frames.append(frame)
        for _ in range(STEPS_PER_FRAME):
            sim.step(cfg.DEFAULT_DT / cfg.SUBSTEPS, cfg.DEFAULT_FLIP_RATIO, COLOR_SPEED)

    path = OUT_DIR / f"{SHAPE_NAMES[shape]}_{suffix}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(path.relative_to(Path.cwd()))


def main() -> None:
    init_taichi()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    window = ti.ui.Window(
        "Lab 2 - 3D FLIP/PIC Fluid GIF Capture",
        (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT),
        vsync=False,
    )
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    for shape in SHAPE_NAMES:
        render_gif(window, canvas, scene, camera, shape, with_obstacle=False)
        render_gif(window, canvas, scene, camera, shape, with_obstacle=True)


if __name__ == "__main__":
    main()
