import pygame
import jax
import jax.numpy as jnp
from functools import partial

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("JAX-optimized Fluid Simulation with String and Particles")

# Simulation parameters
num_points = 20
rest_length = 10
k = 0.9999  # Spring constant
damping = 0.5
fluid_resistance = 0.1
gravity = jnp.array([0, 9.8])
dt = 0.1

# Fluid simulation parameters
resolution = 64
iterations = 16
viscosity = 0.1
diffusion = 0.0001

# Particle parameters
num_particles = 500
particle_size = 2
collision_radius = 5

# Initialize string points
points = jnp.array([(i * width / num_points, height / 2) for i in range(num_points)])
velocities = jnp.zeros_like(points)

# Initialize particles
key = jax.random.PRNGKey(0)
particles = jax.random.uniform(key, (num_particles, 2)) * jnp.array([width, height])
particle_velocities = jnp.zeros_like(particles)

# Initialize fluid velocity field
fluid_field = jnp.zeros((resolution, resolution, 2))

@jax.jit
def apply_boundaries(point):
    return jnp.clip(point, jnp.array([10, 10]), jnp.array([width - 10, height - 10]))

@partial(jax.jit, static_argnums=(3, 4))
def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (resolution - 2) * (resolution - 2)
    return jax.lax.fori_loop(
        0, iterations,
        lambda _, x: (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) / (1 + 4 * a),
        x
    )

@jax.jit
def project(vx, vy):
    div = -0.5 * (vx[2:, 1:-1] - vx[:-2, 1:-1] + vy[1:-1, 2:] - vy[1:-1, :-2]) / resolution
    p = jnp.zeros_like(vx)
    
    def body_fun(_, p):
        return (div + p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]) / 4
    
    p = jax.lax.fori_loop(0, iterations, body_fun, p)
    
    vx[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * resolution
    vy[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * resolution
    return vx, vy

@partial(jax.jit, static_argnums=(2, 3))
def fluid_step(field, visc, dt):
    vx, vy = field[..., 0], field[..., 1]
    vx0, vy0 = vx, vy

    vx = diffuse(1, vx, vx0, visc, dt)
    vy = diffuse(2, vy, vy0, visc, dt)

    vx, vy = project(vx, vy)

    vx0, vy0 = vx, vy
    vx = vx - dt * (vx * jnp.gradient(vx, axis=0) + vy * jnp.gradient(vx, axis=1))
    vy = vy - dt * (vx * jnp.gradient(vy, axis=0) + vy * jnp.gradient(vy, axis=1))

    vx, vy = project(vx, vy)

    return jnp.stack([vx, vy], axis=-1)

@jax.jit
def get_fluid_velocity(x, y, fluid_field):
    i, j = (x * resolution / width).astype(int), (y * resolution / height).astype(int)
    i = jnp.clip(i, 0, resolution - 1)
    j = jnp.clip(j, 0, resolution - 1)
    return fluid_field[j, i]

@jax.jit
def update_string(points, velocities, fluid_field):
    forces = jnp.zeros_like(points)
    for i in range(num_points):
        if i > 0:
            delta = points[i] - points[i-1]
            distance = jnp.linalg.norm(delta)
            force = k * (distance - rest_length) * delta / distance
            forces = forces.at[i].add(-force)
            forces = forces.at[i-1].add(force)
        if i < num_points - 1:
            delta = points[i] - points[i+1]
            distance = jnp.linalg.norm(delta)
            force = k * (distance - rest_length) * delta / distance
            forces = forces.at[i].add(-force)
            forces = forces.at[i+1].add(force)

    accelerations = forces / 1.0 + gravity - fluid_resistance * velocities * jnp.linalg.norm(velocities, axis=1)[:, jnp.newaxis]
    accelerations += jax.vmap(lambda p: get_fluid_velocity(p[0], p[1], fluid_field))(points)

    velocities += accelerations * dt
    velocities *= damping
    points += velocities * dt
    points = jax.vmap(apply_boundaries)(points)

    points = points.at[0].set([width // 4, height // 2])
    points = points.at[-1].set([3 * width // 4, height // 2])

    return points, velocities

@jax.jit
def update_particles(particles, particle_velocities, fluid_field, points):
    fluid_vel = jax.vmap(lambda p: get_fluid_velocity(p[0], p[1], fluid_field))(particles)
    particle_velocities += fluid_vel * dt
    particle_velocities *= damping
    particles += particle_velocities
    particles = jax.vmap(apply_boundaries)(particles)

    def collision_check(particle, velocity):
        for j in range(num_points - 1):
            p1, p2 = points[j], points[j+1]
            closest_point = jnp.array([jnp.clip(particle[0], p1[0], p2[0]), jnp.clip(particle[1], p1[1], p2[1])])
            distance = jnp.linalg.norm(particle - closest_point)
            normal = (particle - closest_point) / (distance + 1e-8)
            particle = jnp.where(distance < collision_radius, closest_point + normal * collision_radius, particle)
            velocity = jnp.where(distance < collision_radius, 
                                 velocity - 2 * jnp.dot(velocity, normal) * normal, 
                                 velocity)
        return particle, velocity

    particles, particle_velocities = jax.vmap(collision_check)(particles, particle_velocities)
    return particles, particle_velocities

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            i, j = int(x * resolution / width), int(y * resolution / height)
            if 0 <= i < resolution - 1 and 0 <= j < resolution - 1:
                strength = 5
                fluid_field = fluid_field.at[j, i].set(jnp.array([event.rel[0] * strength, event.rel[1] * strength]))

    # Clear the screen
    screen.fill((0, 0, 0))

    # Update fluid simulation
    fluid_field = fluid_step(fluid_field, viscosity, dt)

    # Update string and particles
    points, velocities = update_string(points, velocities, fluid_field)
    particles, particle_velocities = update_particles(particles, particle_velocities, fluid_field, points)

    # Draw the string
    for i in range(num_points - 1):
        start = tuple(points[i].astype(int))
        end = tuple(points[i+1].astype(int))
        pygame.draw.line(screen, (255, 255, 255), start, end, 2)

    # Draw particles
    for particle in particles:
        pygame.draw.circle(screen, (0, 255, 255), particle.astype(int), particle_size)

    # Draw boundaries
    pygame.draw.rect(screen, (255, 0, 0), (0, 0, width, height), 2)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()