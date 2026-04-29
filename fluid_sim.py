"""3D PIC/FLIP fluid solver on a staggered MAC grid."""

import taichi as ti

import config as cfg


AIR = 0
FLUID = 1
SOLID = 2

COLOR_SPEED = cfg.COLOR_SPEED
COLOR_DENSITY = cfg.COLOR_DENSITY
COLOR_PRESSURE = cfg.COLOR_PRESSURE


@ti.data_oriented
class FlipFluid3D:
    def __init__(self) -> None:
        self.n = cfg.GRID_RES
        self.h = cfg.DOMAIN_SIZE / cfg.GRID_RES
        self.inv_h = 1.0 / self.h
        self.particle_count = (
            cfg.PARTICLES_X
            * cfg.PARTICLES_Y
            * cfg.PARTICLES_Z
            * cfg.PARTICLES_PER_CELL
        )

        self.p_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.p_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)
        self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_count)

        self.u = ti.field(dtype=ti.f32, shape=(self.n + 1, self.n, self.n))
        self.v = ti.field(dtype=ti.f32, shape=(self.n, self.n + 1, self.n))
        self.w = ti.field(dtype=ti.f32, shape=(self.n, self.n, self.n + 1))
        self.u_old = ti.field(dtype=ti.f32, shape=(self.n + 1, self.n, self.n))
        self.v_old = ti.field(dtype=ti.f32, shape=(self.n, self.n + 1, self.n))
        self.w_old = ti.field(dtype=ti.f32, shape=(self.n, self.n, self.n + 1))
        self.u_weight = ti.field(dtype=ti.f32, shape=(self.n + 1, self.n, self.n))
        self.v_weight = ti.field(dtype=ti.f32, shape=(self.n, self.n + 1, self.n))
        self.w_weight = ti.field(dtype=ti.f32, shape=(self.n, self.n, self.n + 1))

        self.cell_type = ti.field(dtype=ti.i32, shape=(self.n, self.n, self.n))
        self.pressure = ti.field(dtype=ti.f32, shape=(self.n, self.n, self.n))
        self.density = ti.field(dtype=ti.f32, shape=(self.n, self.n, self.n))

        self.obstacle_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.obstacle_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.obstacle_render_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.box_edge_vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)
        self.avg_divergence = ti.field(dtype=ti.f32, shape=())
        self.max_speed = ti.field(dtype=ti.f32, shape=())

        self.reset()

    def reset(self) -> None:
        self.reset_particles()
        self.set_obstacle(*cfg.OBSTACLE_INITIAL, 0.0, 0.0, 0.0)
        self.update_box_edges()
        self.update_obstacle_render()
        self.clear_grid()

    def step(self, dt: float, flip_ratio: float, color_mode: int) -> None:
        self.integrate_particles(dt)
        self.handle_particle_collisions()
        self.clear_grid()
        self.transfer_to_grid()
        self.enforce_solid_faces()
        self.copy_grid_velocities()
        self.solve_incompressibility(cfg.PRESSURE_ITERS, cfg.OVER_RELAXATION)
        self.compute_divergence_metric()
        self.transfer_to_particles(flip_ratio)
        self.handle_particle_collisions()
        self.update_colors(color_mode)
        self.update_obstacle_render()

    def obstacle_numpy(self):
        p = self.obstacle_pos.to_numpy()
        return float(p[0]), float(p[1]), float(p[2])

    def avg_divergence_numpy(self) -> float:
        return float(self.avg_divergence.to_numpy())

    @ti.kernel
    def reset_particles(self):
        start = ti.Vector([0.13, 0.10, 0.20])
        spacing = cfg.PARTICLE_SEPARATION
        for p in self.p_pos:
            local = p // cfg.PARTICLES_PER_CELL
            duplicate = p % cfg.PARTICLES_PER_CELL
            ix = local % cfg.PARTICLES_X
            iy = (local // cfg.PARTICLES_X) % cfg.PARTICLES_Y
            iz = local // (cfg.PARTICLES_X * cfg.PARTICLES_Y)
            stagger = ti.Vector(
                [
                    0.5 * ti.cast(iy % 2, ti.f32),
                    0.0,
                    0.5 * ti.cast((ix + iy) % 2, ti.f32),
                ]
            )
            jitter = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5])
            jitter *= spacing * 0.22
            offset = (ti.Vector([ix, iy, iz]) + stagger) * spacing
            offset += ti.cast(duplicate, ti.f32) * ti.Vector([0.23, 0.17, 0.31]) * spacing
            self.p_pos[p] = start + offset + jitter
            self.p_vel[p] = ti.Vector([0.0, 0.0, 0.0])
            self.p_color[p] = ti.Vector([0.05, 0.48, 1.0])

    @ti.kernel
    def set_obstacle(
        self, x: ti.f32, y: ti.f32, z: ti.f32, vx: ti.f32, vy: ti.f32, vz: ti.f32
    ):
        pos = ti.Vector([x, y, z])
        pos.x = ti.min(ti.max(pos.x, cfg.CONTAINER_MIN + cfg.OBSTACLE_RADIUS), cfg.CONTAINER_MAX - cfg.OBSTACLE_RADIUS)
        pos.y = ti.min(ti.max(pos.y, cfg.CONTAINER_FLOOR + cfg.OBSTACLE_RADIUS), cfg.CONTAINER_CEILING - cfg.OBSTACLE_RADIUS)
        pos.z = ti.min(ti.max(pos.z, cfg.CONTAINER_MIN + cfg.OBSTACLE_RADIUS), cfg.CONTAINER_MAX - cfg.OBSTACLE_RADIUS)
        vel = ti.Vector([vx, vy, vz])
        speed = vel.norm()
        if speed > cfg.OBSTACLE_MAX_SPEED:
            vel *= cfg.OBSTACLE_MAX_SPEED / speed
        self.obstacle_pos[None] = pos
        self.obstacle_vel[None] = vel
        self.obstacle_render_pos[0] = pos

    @ti.kernel
    def integrate_particles(self, dt: ti.f32):
        for p in self.p_pos:
            self.p_vel[p].y += cfg.GRAVITY * dt
            self.p_pos[p] += self.p_vel[p] * dt

    @ti.kernel
    def handle_particle_collisions(self):
        obs = self.obstacle_pos[None]
        obs_vel = self.obstacle_vel[None]
        obs_r = cfg.OBSTACLE_RADIUS + cfg.PARTICLE_RADIUS
        for p in self.p_pos:
            pos = self.p_pos[p]
            vel = self.p_vel[p]
            self.collide_container(pos, vel)
            d = pos - obs
            dist = d.norm()
            if dist < obs_r:
                n = ti.Vector([1.0, 0.0, 0.0])
                if dist > 1.0e-6:
                    n = d / dist
                pos = obs + n * obs_r
                rel = vel - obs_vel
                vn = rel.dot(n)
                if vn < 0.0:
                    vel -= vn * n
            self.p_pos[p] = pos
            self.p_vel[p] = vel

    @ti.kernel
    def clear_grid(self):
        for i, j, k in self.u:
            self.u[i, j, k] = 0.0
            self.u_weight[i, j, k] = 0.0
            self.u_old[i, j, k] = 0.0
        for i, j, k in self.v:
            self.v[i, j, k] = 0.0
            self.v_weight[i, j, k] = 0.0
            self.v_old[i, j, k] = 0.0
        for i, j, k in self.w:
            self.w[i, j, k] = 0.0
            self.w_weight[i, j, k] = 0.0
            self.w_old[i, j, k] = 0.0
        for i, j, k in self.cell_type:
            self.pressure[i, j, k] = 0.0
            self.density[i, j, k] = 0.0
            self.cell_type[i, j, k] = self.initial_cell_type(i, j, k)

    @ti.kernel
    def transfer_to_grid(self):
        for p in self.p_pos:
            pos = self.p_pos[p]
            vel = self.p_vel[p]
            cell = self.world_to_cell(pos)
            if self.cell_type[cell] != SOLID:
                self.cell_type[cell] = FLUID
                ti.atomic_add(self.density[cell], 1.0)
            self.splat_u(pos, vel.x)
            self.splat_v(pos, vel.y)
            self.splat_w(pos, vel.z)
        for i, j, k in self.u:
            if self.u_weight[i, j, k] > 0.0:
                self.u[i, j, k] /= self.u_weight[i, j, k]
        for i, j, k in self.v:
            if self.v_weight[i, j, k] > 0.0:
                self.v[i, j, k] /= self.v_weight[i, j, k]
        for i, j, k in self.w:
            if self.w_weight[i, j, k] > 0.0:
                self.w[i, j, k] /= self.w_weight[i, j, k]
    @ti.kernel
    def copy_grid_velocities(self):
        for i, j, k in self.u:
            self.u_old[i, j, k] = self.u[i, j, k]
        for i, j, k in self.v:
            self.v_old[i, j, k] = self.v[i, j, k]
        for i, j, k in self.w:
            self.w_old[i, j, k] = self.w[i, j, k]

    def solve_incompressibility(self, iterations: int, over_relaxation: float) -> None:
        for _ in range(iterations):
            self.pressure_iteration(over_relaxation)
        self.enforce_solid_faces()

    @ti.kernel
    def pressure_iteration(self, over_relaxation: ti.f32):
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == FLUID:
                sx0 = self.open_cell(i - 1, j, k)
                sx1 = self.open_cell(i + 1, j, k)
                sy0 = self.open_cell(i, j - 1, k)
                sy1 = self.open_cell(i, j + 1, k)
                sz0 = self.open_cell(i, j, k - 1)
                sz1 = self.open_cell(i, j, k + 1)
                s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1
                if s > 0.0:
                    div = (
                        self.u[i + 1, j, k] - self.u[i, j, k]
                        + self.v[i, j + 1, k] - self.v[i, j, k]
                        + self.w[i, j, k + 1] - self.w[i, j, k]
                    )
                    corr = div / s * over_relaxation
                    self.u[i, j, k] += sx0 * corr
                    self.u[i + 1, j, k] -= sx1 * corr
                    self.v[i, j, k] += sy0 * corr
                    self.v[i, j + 1, k] -= sy1 * corr
                    self.w[i, j, k] += sz0 * corr
                    self.w[i, j, k + 1] -= sz1 * corr
                    self.pressure[i, j, k] += corr

    @ti.kernel
    def transfer_to_particles(self, flip_ratio: ti.f32):
        max_speed = 0.0
        for p in self.p_pos:
            pos = self.p_pos[p]
            pic = ti.Vector([
                self.sample_u(pos, False),
                self.sample_v(pos, False),
                self.sample_w(pos, False),
            ])
            delta = ti.Vector([
                self.sample_u(pos, False) - self.sample_u(pos, True),
                self.sample_v(pos, False) - self.sample_v(pos, True),
                self.sample_w(pos, False) - self.sample_w(pos, True),
            ])
            flip = self.p_vel[p] + delta
            vel = pic * (1.0 - flip_ratio) + flip * flip_ratio
            self.p_vel[p] = vel
            max_speed = ti.max(max_speed, vel.norm())
        self.max_speed[None] = max_speed

    @ti.kernel
    def compute_divergence_metric(self):
        total = 0.0
        count = 0
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == FLUID:
                div = (
                    self.u[i + 1, j, k] - self.u[i, j, k]
                    + self.v[i, j + 1, k] - self.v[i, j, k]
                    + self.w[i, j, k + 1] - self.w[i, j, k]
                )
                total += ti.abs(div)
                count += 1
        self.avg_divergence[None] = total / ti.max(1, count)

    @ti.kernel
    def update_colors(self, color_mode: ti.i32):
        max_speed = ti.max(self.max_speed[None], 1.0)
        for p in self.p_pos:
            pos = self.p_pos[p]
            cell = self.world_to_cell(pos)
            speed_t = ti.min(self.p_vel[p].norm() / max_speed, 1.0)
            density_t = ti.min(self.density[cell] / 5.0, 1.0)
            pressure_t = ti.min(ti.abs(self.pressure[cell]) * 0.8, 1.0)
            t = speed_t
            if color_mode == COLOR_DENSITY:
                t = density_t
            elif color_mode == COLOR_PRESSURE:
                t = pressure_t
            deep = ti.Vector([0.03, 0.28, 0.9])
            mid = ti.Vector([0.05, 0.75, 0.95])
            foam = ti.Vector([0.75, 0.95, 1.0])
            col = deep * (1.0 - t * 2.0) + mid * (t * 2.0)
            if t > 0.5:
                a = (t - 0.5) * 2.0
                col = mid * (1.0 - a) + foam * a
            self.p_color[p] = ti.max(ti.Vector([0.0, 0.0, 0.0]), ti.min(col, ti.Vector([1.0, 1.0, 1.0])))

    @ti.kernel
    def update_obstacle_render(self):
        self.obstacle_render_pos[0] = self.obstacle_pos[None]

    @ti.kernel
    def update_box_edges(self):
        mn = cfg.CONTAINER_MIN
        mx = cfg.CONTAINER_MAX
        floor = cfg.CONTAINER_FLOOR
        ceil = cfg.CONTAINER_CEILING
        corners = ti.Matrix.rows([
            [mn, floor, mn], [mx, floor, mn], [mx, floor, mx], [mn, floor, mx],
            [mn, ceil, mn], [mx, ceil, mn], [mx, ceil, mx], [mn, ceil, mx],
        ])
        pairs = ti.Matrix.rows([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])
        for e in range(12):
            a = pairs[e, 0]
            b = pairs[e, 1]
            self.box_edge_vertices[e * 2] = ti.Vector([corners[a, 0], corners[a, 1], corners[a, 2]])
            self.box_edge_vertices[e * 2 + 1] = ti.Vector([corners[b, 0], corners[b, 1], corners[b, 2]])

    @ti.kernel
    def enforce_solid_faces(self):
        for i, j, k in self.u:
            left = i == 0 or self.cell_type[i - 1, j, k] == SOLID
            right = i == self.n or self.cell_type[i, j, k] == SOLID
            if left or right:
                self.u[i, j, k] = 0.0
        for i, j, k in self.v:
            down = j == 0 or self.cell_type[i, j - 1, k] == SOLID
            up = j == self.n or self.cell_type[i, j, k] == SOLID
            if down or up:
                self.v[i, j, k] = 0.0
        for i, j, k in self.w:
            back = k == 0 or self.cell_type[i, j, k - 1] == SOLID
            front = k == self.n or self.cell_type[i, j, k] == SOLID
            if back or front:
                self.w[i, j, k] = 0.0

    @ti.func
    def initial_cell_type(self, i, j, k):
        center = (ti.Vector([i, j, k]) + 0.5) * self.h
        solid = (
            center.x < cfg.CONTAINER_MIN
            or center.x > cfg.CONTAINER_MAX
            or center.y < cfg.CONTAINER_FLOOR
            or center.y > cfg.CONTAINER_CEILING
            or center.z < cfg.CONTAINER_MIN
            or center.z > cfg.CONTAINER_MAX
            or (center - self.obstacle_pos[None]).norm() < cfg.OBSTACLE_RADIUS
        )
        return SOLID if solid else AIR

    @ti.func
    def open_cell(self, i, j, k):
        value = 0.0
        if 0 <= i < self.n and 0 <= j < self.n and 0 <= k < self.n:
            if self.cell_type[i, j, k] != SOLID:
                value = 1.0
        return value

    @ti.func
    def world_to_cell(self, pos):
        idx = ti.cast(ti.floor(pos * self.inv_h), ti.i32)
        return ti.Vector([
            ti.min(ti.max(idx.x, 0), self.n - 1),
            ti.min(ti.max(idx.y, 0), self.n - 1),
            ti.min(ti.max(idx.z, 0), self.n - 1),
        ])

    @ti.func
    def collide_container(self, pos: ti.template(), vel: ti.template()):
        r = cfg.PARTICLE_RADIUS
        if pos.x < cfg.CONTAINER_MIN + r:
            pos.x = cfg.CONTAINER_MIN + r
            vel.x = ti.max(vel.x, 0.0)
        if pos.x > cfg.CONTAINER_MAX - r:
            pos.x = cfg.CONTAINER_MAX - r
            vel.x = ti.min(vel.x, 0.0)
        if pos.y < cfg.CONTAINER_FLOOR + r:
            pos.y = cfg.CONTAINER_FLOOR + r
            vel.y = ti.max(vel.y, 0.0)
        if pos.y > cfg.CONTAINER_CEILING - r:
            pos.y = cfg.CONTAINER_CEILING - r
            vel.y = ti.min(vel.y, 0.0)
        if pos.z < cfg.CONTAINER_MIN + r:
            pos.z = cfg.CONTAINER_MIN + r
            vel.z = ti.max(vel.z, 0.0)
        if pos.z > cfg.CONTAINER_MAX - r:
            pos.z = cfg.CONTAINER_MAX - r
            vel.z = ti.min(vel.z, 0.0)

    @ti.func
    def splat_u(self, pos, value):
        self.splat_component(pos, value, 0)

    @ti.func
    def splat_v(self, pos, value):
        self.splat_component(pos, value, 1)

    @ti.func
    def splat_w(self, pos, value):
        self.splat_component(pos, value, 2)

    @ti.func
    def splat_component(self, pos, value, component: ti.template()):
        g = pos * self.inv_h
        if component == 0:
            g -= ti.Vector([0.0, 0.5, 0.5])
        elif component == 1:
            g -= ti.Vector([0.5, 0.0, 0.5])
        else:
            g -= ti.Vector([0.5, 0.5, 0.0])
        base = ti.cast(ti.floor(g), ti.i32)
        f = g - ti.floor(g)
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            node = base + ti.Vector([ox, oy, oz])
            weight = (1.0 - ti.abs(f.x - ox)) * (1.0 - ti.abs(f.y - oy)) * (1.0 - ti.abs(f.z - oz))
            if component == 0:
                if 0 <= node.x <= self.n and 0 <= node.y < self.n and 0 <= node.z < self.n:
                    ti.atomic_add(self.u[node], value * weight)
                    ti.atomic_add(self.u_weight[node], weight)
            elif component == 1:
                if 0 <= node.x < self.n and 0 <= node.y <= self.n and 0 <= node.z < self.n:
                    ti.atomic_add(self.v[node], value * weight)
                    ti.atomic_add(self.v_weight[node], weight)
            else:
                if 0 <= node.x < self.n and 0 <= node.y < self.n and 0 <= node.z <= self.n:
                    ti.atomic_add(self.w[node], value * weight)
                    ti.atomic_add(self.w_weight[node], weight)

    @ti.func
    def sample_u(self, pos, old: ti.template()):
        return self.sample_component(pos, 0, old)

    @ti.func
    def sample_v(self, pos, old: ti.template()):
        return self.sample_component(pos, 1, old)

    @ti.func
    def sample_w(self, pos, old: ti.template()):
        return self.sample_component(pos, 2, old)

    @ti.func
    def sample_component(self, pos, component: ti.template(), old: ti.template()):
        g = pos * self.inv_h
        max_i = self.n - 1
        max_j = self.n - 1
        max_k = self.n - 1
        if component == 0:
            g -= ti.Vector([0.0, 0.5, 0.5])
            max_i = self.n
        elif component == 1:
            g -= ti.Vector([0.5, 0.0, 0.5])
            max_j = self.n
        else:
            g -= ti.Vector([0.5, 0.5, 0.0])
            max_k = self.n
        g = ti.Vector([
            ti.min(ti.max(g.x, 0.0), ti.cast(max_i, ti.f32) - 1.0e-4),
            ti.min(ti.max(g.y, 0.0), ti.cast(max_j, ti.f32) - 1.0e-4),
            ti.min(ti.max(g.z, 0.0), ti.cast(max_k, ti.f32) - 1.0e-4),
        ])
        base = ti.cast(ti.floor(g), ti.i32)
        f = g - ti.floor(g)
        acc = 0.0
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            node = ti.Vector([
                ti.min(base.x + ox, max_i),
                ti.min(base.y + oy, max_j),
                ti.min(base.z + oz, max_k),
            ])
            weight = (1.0 - ti.abs(f.x - ox)) * (1.0 - ti.abs(f.y - oy)) * (1.0 - ti.abs(f.z - oz))
            if component == 0:
                acc += (self.u_old[node] if old else self.u[node]) * weight
            elif component == 1:
                acc += (self.v_old[node] if old else self.v[node]) * weight
            else:
                acc += (self.w_old[node] if old else self.w[node]) * weight
        return acc
