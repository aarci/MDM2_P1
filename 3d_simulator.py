import numpy as np
import perlin_numpy
import matplotlib.pyplot as plt

grid_size = (10, 10, 10)


perlin_noise = perlin_numpy.generate_perlin_noise_3d(grid_size, (1, 1, 1))


y, x, z = np.meshgrid(np.linspace(0, 1, grid_size[0], endpoint=False),
                      np.linspace(0, 1, grid_size[1], endpoint=False),
                      np.linspace(0, 1, grid_size[2], endpoint=False))


dx, dy, dz = np.gradient(perlin_noise)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.quiver(x, y, z, dx, dy, dz, length=0.05, normalize=True)
import numpy as np
import perlin_numpy
import matplotlib.pyplot as plt

grid_size = (10, 10, 10)


perlin_noise = perlin_numpy.generate_perlin_noise_3d(grid_size, (1, 1, 1))


y, x, z = np.meshgrid(np.linspace(0, 1, grid_size[0], endpoint=False),
                      np.linspace(0, 1, grid_size[1], endpoint=False),
                      np.linspace(0, 1, grid_size[2], endpoint=False))


dx, dy, dz = np.gradient(perlin_noise)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.quiver(x, y, z, dx, dy, dz, length=0.05, normalize=True)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Wind Direction Field')


point_a = np.array([0.2, 0.2, 0.2])
point_b = np.array([0.8, 0.8, 0.8])
fixed_points = np.array([point_a, point_b])
ax.scatter(fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2], color='red', s=50, label='Fixed Points')


num_steps = 200
point = point_a.copy()
trajectory = [point]
thrust_strength = 0.05

for _ in range(num_steps):

    direction_to_b = point_b - point
    if np.linalg.norm(direction_to_b) < 0.05:
        break
    direction_to_b = direction_to_b / np.linalg.norm(direction_to_b) * 0.05  # Normalize and scale the step size


    ix = min(max(int(round(point[0] * grid_size[0])), 0), grid_size[0] - 1)
    iy = min(max(int(round(point[1] * grid_size[1])), 0), grid_size[1] - 1)
    iz = min(max(int(round(point[2] * grid_size[2])), 0), grid_size[2] - 1)


    wind_direction = np.array([dx[ix, iy, iz], dy[ix, iy, iz], dz[ix, iy, iz]])


    if np.linalg.norm(wind_direction) != 0:
        wind_direction = wind_direction / np.linalg.norm(wind_direction) * 0.1  # Reduce wind effect
    resultant_direction = direction_to_b + thrust_strength * wind_direction
    resultant_direction = resultant_direction / np.linalg.norm(
        resultant_direction) * 0.05  # Normalize and scale the step size


    point = point + resultant_direction
    trajectory.append(point)

trajectory = np.array(trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue', label='Path from A to B')


ax.legend()

plt.show()
