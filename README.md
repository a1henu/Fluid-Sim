# Lab 2 Fluid Simulation

Taichi implementation of a 3D PIC/FLIP particle fluid demo on a staggered MAC
grid. The solver follows the Lab 2 reference structure:

1. integrate particles
2. collide with tank and obstacle
3. separate nearby particles with a spatial hash
4. transfer particle velocity to MAC grid
5. mark solid / fluid / air cells and estimate density
6. solve incompressibility with Gauss-Seidel projection
7. transfer velocities back with PIC/FLIP blending

## Run

```bash
uv sync
uv run python main.py
```

## Controls

The demo starts paused so the spherical obstacle can be placed before simulation.

- `Space`: start / pause / resume
- Left mouse drag: place the obstacle in the tank
- Right mouse drag: orbit the camera
- `Q` / `E`: zoom in / out
- `[` / `]`: change time step
- `,` / `.`: adjust `flipRatio`
- `1`: PIC, `flipRatio = 0`
- `2`: FLIP95, `flipRatio = 0.95`
- `3`: FLIP, `flipRatio = 1`
- `C`: switch particle coloring by speed / density / pressure
- `R`: reset

## Validation

```bash
uv run python -m py_compile config.py fluid_sim.py main.py
uv run python -c "import taichi as ti; ti.init(arch=ti.cpu); import config as cfg; from fluid_sim import FlipFluid3D, COLOR_SPEED; sim=FlipFluid3D(); [sim.step(cfg.DEFAULT_DT/cfg.SUBSTEPS, cfg.DEFAULT_FLIP_RATIO, COLOR_SPEED) for _ in range(4)]; print(sim.avg_divergence_numpy())"
```
