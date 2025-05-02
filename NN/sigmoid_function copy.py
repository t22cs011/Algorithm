import numpy as np
import matplotlib.pyplot as plt

# シグモイド関数の定義
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# シグモイド関数の導関数の定義
def sigmoid_derivative(z):
    sigmoid = sigmoid_function(z)
    return sigmoid * (1 - sigmoid)

# z の範囲を設定
z_values = np.linspace(-10, 10, 500)  # -10 から 10 までの範囲を設定
f_values = sigmoid_function(z_values)
f_derivative_values = sigmoid_derivative(z_values)

# グラフの作成
plt.figure(figsize=(8, 6))
plt.plot(z_values, f_values, color='b', label=r'$f(z)=\frac{1}{1+e^{-z}}$')  # シグモイド関数
plt.plot(z_values, f_derivative_values, color='r', linestyle='--', label=r"$f'(z)$")  # 導関数

# ラベルの設定
plt.axhline(0, color='black', linewidth=0.8)  # 横軸
plt.axvline(0, color='black', linewidth=0.8)  # 縦軸
plt.xlabel("z")  # 横軸ラベル
plt.ylabel("Value")  # 縦軸ラベル

# グラフの装飾
plt.title("Sigmoid Function and Its Derivative")
plt.legend()
plt.grid(True)
plt.tight_layout()

# グラフを画像として保存
plt.savefig("sigmoid_function_and_derivative_plot.png")  # ファイル名を指定

# グラフを表示
plt.show()
