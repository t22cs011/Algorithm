import numpy as np

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr  # η (学習率)
        self.h = None  # 各要素の勾配二乗和 h を保持
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)  # h₀ = 0 で初期化
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]  # h ← h + (∂L/∂W)⊙(∂L/∂W)    (式6.5)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # W ← W - η * (1/√h) ⊙ ∂L/∂W    (式6.6)
            # 1e-7はself.h[key]の中に0があった場合, ゼロ除算を避けるために加算
