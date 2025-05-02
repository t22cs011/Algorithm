import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 関数 f(x, y) = 0.5 * (x^2 + 10*y^2) とその勾配
def f(x, y):
    return 0.5 * (x**2 + 10 * y**2)

def grad_f(x, y):
    return np.array([x, 10 * y])

# 勾配降下法 with モーメンタムのシミュレーション
def gradient_descent_momentum(alpha, eta, init, num_iters):
    x, y = init
    v = np.array([0.0, 0.0])
    traj = [(x, y)]
    for _ in range(num_iters):
        grad = grad_f(x, y)
        v = alpha * v - eta * grad
        x, y = np.array([x, y]) + v
        traj.append((x, y))
    return np.array(traj)

# シミュレーションパラメータ
eta = 0.1          # 学習率
num_iters = 60     # 更新回数
init = np.array([8.0, 8.0])  # 初期位置

# 異なるモーメンタム係数
alphas = [0.0, 0.5, 0.9]
trajectories = {alpha: gradient_descent_momentum(alpha, eta, init, num_iters)
                for alpha in alphas}

# 描画用のメッシュグリッドを作成（関数の等高線表示用）
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# プロット初期化
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=30, cmap='viridis')
colors = {0.0: 'red', 0.5: 'blue', 0.9: 'green'}
lines = {}
points = {}

for alpha in alphas:
    # 各軌跡の線を初期化
    line, = ax.plot([], [], color=colors[alpha], label=f'α = {alpha}', lw=2)
    lines[alpha] = line
    # 現在の位置を示す点
    point, = ax.plot([], [], 'o', color=colors[alpha])
    points[alpha] = point

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Descent with Momentum')
ax.legend()

# アニメーション更新関数
def animate(i):
    for alpha in alphas:
        traj = trajectories[alpha]
        # i番目までの軌跡を更新
        lines[alpha].set_data(traj[:i+1, 0], traj[:i+1, 1])
        # 現在の位置の点を更新（スカラー値をリストに変換）
        points[alpha].set_data([traj[i, 0]], [traj[i, 1]])
    ax.set_title(f'Iteration {i}')
    return list(lines.values()) + list(points.values())

# アニメーション作成
anim = FuncAnimation(fig, animate, frames=num_iters, interval=200, blit=True)

# GIFとして保存（PillowWriterを使用）
anim.save("momentum_comparison.gif", writer=PillowWriter(fps=5))

plt.show()
