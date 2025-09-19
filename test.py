import numpy as np
import matplotlib.pyplot as plt

# Parameters for Case 3
L = 1.0
H = 0.5
Nx = 40
Ny = 20

Dx = L / (Nx - 1)
Dy = H / (Ny - 1)

k = 16 * (np.linspace(0, H, Ny) / H + 1)
k = np.tile(k.reshape(Ny, 1), (1, Nx))

S = -1.5 * np.ones((Ny, Nx))

T1 = 15  # left boundary temperature
q_bottom = 5000  # bottom boundary heat flux
T3 = 10  # right boundary temperature

T = np.zeros((Ny, Nx))
T[:, 0] = T1
T[:, -1] = T3

max_iter = 10000
tolerance = 1e-6

aw = np.zeros_like(T)
ae = np.zeros_like(T)
an = np.zeros_like(T)
as_ = np.zeros_like(T)
ap = np.zeros_like(T)

for j in range(1, Ny-1):
    for i in range(1, Nx-1):
        k_e = (k[j, i] + k[j, i+1]) / 2
        k_w = (k[j, i] + k[j, i-1]) / 2
        k_n = (k[j, i] + k[j-1, i]) / 2
        k_s = (k[j, i] + k[j+1, i]) / 2

        ae[j,i] = k_e / Dx**2
        aw[j,i] = k_w / Dx**2
        an[j,i] = k_n / Dy**2
        as_[j,i] = k_s / Dy**2

        ap[j,i] = ae[j,i] + aw[j,i] + an[j,i] + as_[j,i]

for iteration in range(max_iter):
    T_old = T.copy()

    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            Sp = S[j,i]
            T[j,i] = (ae[j,i]*T[j,i+1] + aw[j,i]*T[j,i-1] + an[j,i]*T[j-1,i] + as_[j,i]*T[j+1,i] - Sp) / ap[j,i]

    # Top boundary Neumann BC: zero gradient
    T[0,:] = T[1,:]

    # Bottom boundary Neumann BC: heat flux q
    T[-1,:] = T[-2,:] + q_bottom * Dy / k[-1,:]

    error = np.linalg.norm(T - T_old, ord=np.inf)
    if error < tolerance:
        break

print(f'Converged in {iteration+1} iterations with error {error:.2e}')

x = np.linspace(0, L, Nx)
y = np.linspace(0, H, Ny)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, T, 50, cmap='inferno')
plt.colorbar(label='Temperature')
plt.title('Temperature Contour for Case 3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
