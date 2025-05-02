import math
import matplotlib.pyplot as plt

def sigmoid_function(y, epsilon):
    # シグモイド関数を定義通りに計算
    return 1 / (1 + math.exp(-epsilon * y))


if __name__ == "__main__":
    # εの値を設定
    epsilon_values = [0.5, 1.0, 5.0]

    # y の範囲を設定
    y_values = [i * 0.1 for i in range(-50, 50)]  # -5.0 から 5.0 まで 0.1 刻み

    # グラフをプロット
    plt.figure(figsize=(8, 6))
    for epsilon in epsilon_values:
        outputs = [sigmoid_function(y, epsilon) for y in y_values]
        plt.plot(y_values, outputs, label=f"ε = {epsilon}")

    plt.xlabel("Input Sum (y)")
    plt.ylabel("Output")
    plt.title("Sigmoid Function Output for Different ε Values")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("sigmoid_function_multiple_epsilon.png")  # グラフを画像として保存
    plt.show()
