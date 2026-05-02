"""Generate report GIFs for the four initial fluid shapes."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from fluid_sim import COLOR_SPEED, FlipFluid3D


OUT_DIR = Path(__file__).resolve().parent / "results"
FRAME_SIZE = 320
FRAME_COUNT = 30
STEPS_PER_FRAME = 3
PARTICLE_STRIDE = 2
WATER_LIFT = 0.25
GIF_DURATION_MS = 70
CAMERA_YAW = 0.78
CAMERA_PITCH = 0.34
CAMERA_DISTANCE = 1.75
CAMERA_TARGET = np.array([0.5, 0.45, 0.5], dtype=np.float32)
CAMERA_FOV_DEG = 45.0

SHAPES = {
    cfg.INIT_SHAPE_CUBE: "cube",
    cfg.INIT_SHAPE_SPHERE: "sphere",
    cfg.INIT_SHAPE_HEART: "heart",
    cfg.INIT_SHAPE_BUNNY: "bunny",
}


def camera_position() -> np.ndarray:
    cp = math.cos(CAMERA_PITCH)
    return np.array(
        [
            CAMERA_TARGET[0] + CAMERA_DISTANCE * math.sin(CAMERA_YAW) * cp,
            CAMERA_TARGET[1] + CAMERA_DISTANCE * math.sin(CAMERA_PITCH),
            CAMERA_TARGET[2] + CAMERA_DISTANCE * math.cos(CAMERA_YAW) * cp,
        ],
        dtype=np.float32,
    )


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = camera_position()
    forward = CAMERA_TARGET - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return eye, right, up, forward


def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye, right, up, forward = camera_basis()
    rel = points - eye[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward
    focal = 1.0 / math.tan(math.radians(CAMERA_FOV_DEG) * 0.5)
    sx = FRAME_SIZE * (0.5 + 0.5 * focal * cam_x / cam_z)
    sy = FRAME_SIZE * (0.5 - 0.5 * focal * cam_y / cam_z)
    valid = cam_z > 1.0e-4
    return sx, sy, cam_z, valid


def draw_container(draw: ImageDraw.ImageDraw):
    mn = cfg.CONTAINER_MIN
    mx = cfg.CONTAINER_MAX
    floor = cfg.CONTAINER_FLOOR
    ceil = cfg.CONTAINER_CEILING
    corners = np.array(
        [
            [mn, floor, mn],
            [mx, floor, mn],
            [mx, floor, mx],
            [mn, floor, mx],
            [mn, ceil, mn],
            [mx, ceil, mn],
            [mx, ceil, mx],
            [mn, ceil, mx],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    sx, sy, _, valid = project(corners)
    for a, b in edges:
        if valid[a] and valid[b]:
            draw.line((sx[a], sy[a], sx[b], sy[b]), fill=(84, 126, 156), width=2)


def draw_obstacle(draw: ImageDraw.ImageDraw, sim: FlipFluid3D):
    obs = np.array(sim.obstacle_numpy(), dtype=np.float32)[None, :]
    sx, sy, depth, valid = project(obs)
    if not valid[0]:
        return
    focal = 1.0 / math.tan(math.radians(CAMERA_FOV_DEG) * 0.5)
    radius = max(3, int(cfg.OBSTACLE_RADIUS * FRAME_SIZE * 0.5 * focal / depth[0]))
    cx = int(sx[0])
    cy = int(sy[0])
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(238, 231, 198),
        outline=(112, 102, 72),
    )


def draw_frame(sim: FlipFluid3D, title: str) -> Image.Image:
    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (13, 18, 26))
    draw = ImageDraw.Draw(image)

    draw.text((18, 12), title, fill=(220, 232, 242))
    draw_container(draw)
    draw_obstacle(draw, sim)

    pos = sim.p_pos.to_numpy()[::PARTICLE_STRIDE]
    colors = (np.clip(sim.p_color.to_numpy()[::PARTICLE_STRIDE], 0.0, 1.0) * 255).astype(
        np.uint8
    )
    sx, sy, depth, valid = project(pos)
    order = np.argsort(-depth)
    focal = 1.0 / math.tan(math.radians(CAMERA_FOV_DEG) * 0.5)

    for idx in order:
        if not valid[idx]:
            continue
        x = int(sx[idx])
        y = int(sy[idx])
        radius = max(1, int(cfg.PARTICLE_RADIUS * FRAME_SIZE * 0.5 * focal / depth[idx]))
        if -4 <= x < FRAME_SIZE + 4 and -4 <= y < FRAME_SIZE + 4:
            shade = max(0.58, min(1.0, 1.15 - 0.18 * depth[idx]))
            color = tuple(int(c * shade) for c in colors[idx])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return image


def render_gif(shape: int, with_obstacle: bool):
    sim = FlipFluid3D(shape)
    pos = sim.p_pos.to_numpy()
    pos[:, 1] += WATER_LIFT
    sim.p_pos.from_numpy(pos)
    if with_obstacle:
        sim.set_obstacle(0.36, 0.32, 0.50, 0.0, 0.0, 0.0)
        suffix = "obstacle"
        title = f"{SHAPES[shape]} hitting obstacle"
    else:
        sim.set_obstacle(0.90, 0.88, 0.90, 0.0, 0.0, 0.0)
        suffix = "normal"
        title = f"{SHAPES[shape]} falling"

    frames: list[Image.Image] = []
    for _ in range(FRAME_COUNT):
        frames.append(draw_frame(sim, title))
        for _ in range(STEPS_PER_FRAME):
            sim.step(cfg.DEFAULT_DT / cfg.SUBSTEPS, cfg.DEFAULT_FLIP_RATIO, COLOR_SPEED)

    path = OUT_DIR / f"{SHAPES[shape]}_{suffix}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=True,
    )
    print(path.relative_to(Path.cwd()))


def main():
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for shape in SHAPES:
        render_gif(shape, with_obstacle=False)
        render_gif(shape, with_obstacle=True)


if __name__ == "__main__":
    main()
