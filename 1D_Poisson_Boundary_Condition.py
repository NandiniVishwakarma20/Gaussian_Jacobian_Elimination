N = 50
phi_0 = 0
phi_1 = 1
epsilon = 1
rho = 1
h = 1 / N
x = np.linspace(0,1,N+1)

n = N-1
A = np.zeros((n,n))
for i in range(n):
    A[i,i] = 2
    if i > 0:
        A[i,i-1] = -1
    if i < n - 1:
        A[i,i+1] = -1

def B(rho_func):
    b = np.array([(h**2 / epsilon) * rho_func(x[i+1]) for i in range(n)], dtype=float)
    b[0] += phi_0
    b[-1] += phi_1
    return b

def rho_const(x): return rho
def rho_0(x): return 0

b_const = B(rho_const)
b_0 = B(rho_0)

gaussian_const = gaussian_elimination(A,b_const)
gaussian_zero = gaussian_elimination(A,b_0)

jacobi_const = jacobi_elimination(A,b_const)
jacobi_zero = jacobi_elimination(A,b_0)

phi_gauss_const = np.concatenate(([phi_0], gaussian_const, [phi_1]))
phi_gauss_zero  = np.concatenate(([phi_0], gaussian_zero, [phi_1]))
phi_jacobi_const = np.concatenate(([phi_0], jacobi_const, [phi_1]))
phi_jacobi_zero  = np.concatenate(([phi_0], jacobi_zero, [phi_1]))

R = rho

phi_const = -R/(2*epsilon) * x**2 + (R/(2*epsilon) + (phi_1 - phi_0)) * x + phi_0
phi0 = phi_0 + (phi_1 - phi_0) * x

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x,phi_gauss_const)
plt.plot(x, phi_jacobi_const,'--')
plt.plot(x, phi_const)

# plt.legend()
# plt.show

plt.subplot(1,2,2)
plt.plot(x, phi_gauss_zero)
plt.plot(x,phi_jacobi_zero, '--')
plt.plot(x, phi0)

# plt.tight_layout()
# plt.legend()
# plt.show()