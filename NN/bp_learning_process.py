import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# ConvergenceWarning を無視
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. データ生成 (非線形: 半月型)
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# 2. MLPClassifier の初期化 (1エポックずつ学習し重みを保持)
mlp = MLPClassifier(
    hidden_layer_sizes=(40,),
    activation='logistic',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=1,         # 1エポックだけ実行
    warm_start=True,    # 前回の重みを保持
    random_state=42
)

# 3. 決定境界描画用メッシュ生成
x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# 4. 学習過程の記録
epochs = 100
boundaries = []
losses = []

for epoch in range(epochs):
    mlp.fit(X, y)
    losses.append(mlp.loss_)
    Z = mlp.predict(grid).reshape(xx.shape)
    boundaries.append(Z)

# 5. アニメーションの準備
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 左: 決定境界プロット
ax1.set_title("Decision Boundary")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# 右: 損失推移プロット
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
line_loss, = ax2.plot([], [], lw=2)

# 更新関数
def animate(i):
    ax1.clear()
    ax1.contourf(xx, yy, boundaries[i], alpha=0.3, cmap='coolwarm')
    ax1.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')
    ax1.set_title(f"Epoch {i+1}/{epochs}")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    # 損失曲線更新
    line_loss.set_data(range(i+1), losses[:i+1])
    ax2.clear()
    ax2.plot(range(i+1), losses[:i+1], lw=2, color='blue')
    ax2.set_xlim(0, epochs)
    ax2.set_ylim(min(losses), max(losses))
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    return ax1.collections + [line_loss]

# アニメーション作成
anim = FuncAnimation(fig, animate, frames=epochs, interval=200, blit=True)

# GIFとして保存
anim.save("mlp_moons_training.gif", writer=PillowWriter(fps=5))

plt.show()
