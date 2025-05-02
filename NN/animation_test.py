import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. データ生成 ---
np.random.seed(0)
# 100点の2次元データ
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 真の境界は x1 + x2 = 0

# --- 2. ロジスティック回帰モデルの初期化 ---
# 重みとバイアスを初期化（ランダムに初期化）
w = np.random.randn(2)
b = np.random.randn(1)[0]
learning_rate = 0.1
num_epochs = 100

# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 3. 誤差逆伝播（勾配降下法）で重み更新 ---
# エポック毎にパラメータの履歴を保存するためのリスト
w_history = []
b_history = []

for epoch in range(num_epochs):
    # 予測
    z = np.dot(X, w) + b
    predictions = sigmoid(z)
    
    # 誤差（クロスエントロピーの勾配を利用）
    error = predictions - y  # 形状：(100,)
    
    # 重みの勾配とバイアスの勾配（平均勾配）
    grad_w = np.dot(X.T, error) / X.shape[0]
    grad_b = np.mean(error)
    
    # パラメータ更新
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    # 履歴に保存
    w_history.append(w.copy())
    b_history.append(b)

# --- 4. アニメーション描画 ---
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
line, = ax.plot([], [], 'k--', linewidth=2)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title("Decision Boundary Evolution via Gradient Descent")

def animate(i):
    # 現在のパラメータ
    w_current = w_history[i]
    b_current = b_history[i]
    # 決定境界: sigmoid(z)=0.5 となる z=0 のときの境界 → w1*x1 + w2*x2 + b = 0
    x_vals = np.linspace(-3, 3, 100)
    # w_current[1] で割る。ゼロ割り回避のため小さな定数を足す
    y_vals = -(w_current[0]*x_vals + b_current) / (w_current[1] + 1e-6)
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {i+1}/{num_epochs}")
    return line,

anim = FuncAnimation(fig, animate, frames=num_epochs, interval=200)

# GIFとして保存
anim.save("boundary_learning.gif", writer=PillowWriter(fps=5))
# MP4の場合（ffmpegがインストールされている必要があります）
# anim.save("boundary_learning.mp4", writer='ffmpeg')

plt.show()
