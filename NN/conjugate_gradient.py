import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock関数と勾配
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
         200 * (x[1] - x[0]**2)
    ])

# Armijo のバックトラック線形探索
def line_search(x, p):
    t = 1.0
    c = 1e-4
    rho = 0.5
    fx = f(x)
    gx = grad(x)
    while f(x + t * p) > fx + c * t * np.dot(gx, p):
        t *= rho
    return t

# 最急降下法 (Armijo)
def steepest_descent(x0, max_iter=50):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(max_iter):
        p = -grad(x)
        alpha = line_search(x, p)
        x = x + alpha * p
        xs.append(x.copy())
    return np.array(xs)

# 非線形共役勾配法 (Fletcher–Reeves + Armijo)
def nonlinear_cg(x0, max_iter=50):
    xs = [x0.copy()]
    x = x0.copy()
    g = grad(x)
    p = -g
    for _ in range(max_iter):
        alpha = line_search(x, p)
        x = x + alpha * p
        g_new = grad(x)
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        p = -g_new + beta * p
        g = g_new
        xs.append(x.copy())
    return np.array(xs)

# 初期点
x0 = np.array([-1.5, 1.5])
sd_path = steepest_descent(x0, max_iter=1000)  # イテレーション回数を1000回に設定
cg_path = nonlinear_cg(x0)

# 等高線描画用グリッド
xs = np.linspace(-2, 2, 400)
ys = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(xs, ys)
Z = 100 * (Y - X**2)**2 + (1 - X)**2

# プロット
plt.figure()
# 等高線を内側に書くために、levels を Z の低い値に集中させる
plt.contour(X, Y, Z, levels=np.linspace(0, 500, 10), colors='black')  # 内側の等高線を強調
plt.plot(sd_path[:,0], sd_path[:,1], marker='o', label='Steepest Descent')
plt.plot(cg_path[:,0], cg_path[:,1], marker='x', label='Nonlinear Conjugate Gradient')
plt.scatter(1, 1, color='red', label='Minimum (1, 1)', zorder=5)
plt.text(1.1, 1, '(1, 1)', color='red')
plt.title('Trajectories on Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.gca().set_aspect('equal')
plt.savefig('conjugate_gradient.png')
plt.show()