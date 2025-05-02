import matplotlib.pyplot as plt

def step_function(y, h):
    print(f"Total potential (y): {y}")
    print(f"Threshold (h): {h}")

    if y >= h:
        print("y >= h, returning 1")
        return 1
    else:
        print("y < h, returning 0")
        return 0


if __name__ == "__main__":
    h = 2.5  # 閾値

    # y の範囲を設定
    y_values = [i * 0.1 for i in range(-50, 50)]  # -5.0 から 5.0 まで 0.1 刻み
    outputs = [step_function(y, h) for y in y_values]

    # グラフをプロット
    plt.figure(figsize=(8, 6))
    plt.plot(y_values, outputs, marker='o', linestyle='-', color='b', label='Step Function Output')
    plt.xlabel("Input Sum (y)")
    plt.ylabel("Output")
    plt.title("Step Function Output")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("step_function_output_plot.png")  # グラフを画像として保存
    plt.show()