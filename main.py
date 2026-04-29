"""3D interactive FLIP/PIC fluid demo."""

from __future__ import annotations

import math

import taichi as ti

import config as cfg
from fluid_sim import COLOR_DENSITY, COLOR_PRESSURE, COLOR_SPEED, FlipFluid3D


COLOR_NAMES = {
    COLOR_SPEED: "speed",
    COLOR_DENSITY: "density",
    COLOR_PRESSURE: "pressure",
}


def init_taichi() -> None:
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def screen_to_obstacle(cursor: tuple[float, float]) -> tuple[float, float, float]:
    x = cfg.CONTAINER_MIN + cursor[0] * (cfg.CONTAINER_MAX - cfg.CONTAINER_MIN)
    z = cfg.CONTAINER_MIN + cursor[1] * (cfg.CONTAINER_MAX - cfg.CONTAINER_MIN)
    return x, cfg.OBSTACLE_PLANE_Y, z


def camera_pos(yaw: float, pitch: float, distance: float) -> tuple[float, float, float]:
    target = (0.5, 0.45, 0.5)
    cp = math.cos(pitch)
    return (
        target[0] + distance * math.sin(yaw) * cp,
        target[1] + distance * math.sin(pitch),
        target[2] + distance * math.cos(yaw) * cp,
    )


def main() -> None:
    init_taichi()
    sim = FlipFluid3D()

    window = ti.ui.Window(
        "Lab 2 - 3D FLIP/PIC Fluid",
        (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    dt = cfg.DEFAULT_DT
    flip_ratio = cfg.DEFAULT_FLIP_RATIO
    color_mode = COLOR_SPEED
    started = False
    paused = True
    yaw = 0.78
    pitch = 0.34
    distance = 1.75
    last_cursor = window.get_cursor_pos()
    last_obstacle = sim.obstacle_numpy()

    while window.running:
        cursor = window.get_cursor_pos()

        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "r":
                sim.reset()
                started = False
                paused = True
                last_obstacle = sim.obstacle_numpy()
            elif event.key == " ":
                if not started:
                    started = True
                    paused = False
                else:
                    paused = not paused
            elif event.key == "[":
                dt = clamp(dt * 0.8, cfg.MIN_DT, cfg.MAX_DT)
            elif event.key == "]":
                dt = clamp(dt * 1.25, cfg.MIN_DT, cfg.MAX_DT)
            elif event.key == ",":
                flip_ratio = clamp(flip_ratio - 0.05, 0.0, 1.0)
            elif event.key == ".":
                flip_ratio = clamp(flip_ratio + 0.05, 0.0, 1.0)
            elif event.key == "1":
                flip_ratio = 0.0
            elif event.key == "2":
                flip_ratio = 0.95
            elif event.key == "3":
                flip_ratio = 1.0
            elif event.key == "c":
                color_mode = (color_mode + 1) % 3

        if window.is_pressed(ti.ui.RMB):
            dx = cursor[0] - last_cursor[0]
            dy = cursor[1] - last_cursor[1]
            yaw -= dx * 4.0
            pitch = clamp(pitch + dy * 3.0, -0.15, 1.15)
        if window.is_pressed("q"):
            distance = clamp(distance * 0.985, 1.05, 2.7)
        if window.is_pressed("e"):
            distance = clamp(distance * 1.015, 1.05, 2.7)

        if window.is_pressed(ti.ui.LMB):
            x, y, z = screen_to_obstacle(cursor)
            frame_dt = dt * cfg.SUBSTEPS
            vx = (x - last_obstacle[0]) / max(frame_dt, 1.0e-6)
            vy = (y - last_obstacle[1]) / max(frame_dt, 1.0e-6)
            vz = (z - last_obstacle[2]) / max(frame_dt, 1.0e-6)
            sim.set_obstacle(x, y, z, vx, vy, vz)
        else:
            sim.set_obstacle(*last_obstacle, 0.0, 0.0, 0.0)
        last_obstacle = sim.obstacle_numpy()
        last_cursor = cursor

        if started and not paused:
            sub_dt = dt / cfg.SUBSTEPS
            for _ in range(cfg.SUBSTEPS):
                sim.step(sub_dt, flip_ratio, color_mode)

        camera.position(*camera_pos(yaw, pitch, distance))
        camera.lookat(0.5, 0.45, 0.5)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(45)

        scene.set_camera(camera)
        scene.ambient_light((0.38, 0.42, 0.48))
        scene.point_light(pos=(0.35, 1.55, 1.1), color=(1.0, 1.0, 1.0))
        scene.lines(sim.box_edge_vertices, width=2.2, color=(0.78, 0.9, 1.0))
        scene.particles(sim.p_pos, radius=cfg.PARTICLE_RADIUS, per_vertex_color=sim.p_color)
        scene.particles(sim.obstacle_render_pos, radius=cfg.OBSTACLE_RADIUS, color=(0.95, 0.95, 0.88))
        canvas.scene(scene)

        window.GUI.begin("3D FLIP Fluid", 0.015, 0.015, 0.32, 0.29)
        window.GUI.text(f"dt: {dt:.5f}  ([ / ])")
        window.GUI.text(f"flipRatio: {flip_ratio:.2f}  (, / . or 1/2/3)")
        window.GUI.text(f"color: {COLOR_NAMES[color_mode]}  (C)")
        window.GUI.text(f"avg divergence: {sim.avg_divergence_numpy():.5f}")
        if not started:
            window.GUI.text("SETUP: drag obstacle, Space to start")
        else:
            window.GUI.text("RUNNING" if not paused else "PAUSED")
        window.GUI.text("LMB: place obstacle    RMB: orbit")
        window.GUI.text("Q/E: zoom    R: reset")
        window.GUI.end()

        window.show()


if __name__ == "__main__":
    main()
