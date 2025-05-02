import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
epsilon = 1
u = np.linspace(-5, 5, 400)
f = np.exp(-epsilon * u)

# グラフの作成
plt.figure(figsize=(6, 4))
plt.plot(u, f, label=r'$f(u)=e^{-\varepsilon u}$')
plt.xlabel('u')
plt.ylabel('f(u)')
plt.title(r'グラフ: $f(u)=\exp(-\varepsilon u)$')  # 修正: raw文字列を使用
plt.legend()
plt.grid(True)

# グラフをファイルとして保存
plt.savefig("exp_function_plot.png")  # ファイル名を指定

# グラフを表示
plt.show()
