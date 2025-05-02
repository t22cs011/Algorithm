import matplotlib.pyplot as plt

def piecewise_linear_function(y, u, a, b):
    if y <= -b / a:
        return 0
    elif -b / a < y < (1 - b) / a:
        return a * y + b
    else:
        return 1


if __name__ == "__main__":
    u = 0.0
    a = 1.0
    b = 0.2

    # y の範囲を設定
    y_values = [i * 0.1 for i in range(-50, 50)]  # -5.0 から 5.0 まで 0.1 刻み
    outputs = [piecewise_linear_function(y, u, a, b) for y in y_values]

    # グラフをプロット
    plt.figure(figsize=(8, 6))
    plt.plot(y_values, outputs, marker='o', linestyle='-', color='b', label='Output')
    plt.xlabel("Input Sum (y)")
    plt.ylabel("Output")
    plt.title("Piecewise Linear Function Output")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("output_plot_y_vs_output.png")
    plt.show()