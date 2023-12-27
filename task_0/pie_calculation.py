import numpy as np

border_length = 2

N = 10**6
points = np.random.uniform(-1, 1, size=[N, 2])

print(points)
poi = [[-1, -1]]
_C = []
for point in points:
    if (point[0] ** 2 + point[1] ** 2) <= (border_length / 2) ** 2:
        _C.append(point)
C = len(_C)
print(C / N)
ratio = C / N
surface_square = border_length**2
pi = surface_square * ratio
print(pi)
