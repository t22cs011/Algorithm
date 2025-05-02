import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr   # η (学習率)
        self.momentum = momentum # α (モーメンタム係数)
        self.v = None   # v_t を保持する辞書 (各パラメータごとの速度)

    def update(self, params, grads):
        """
        params (dict): パラメータ W を格納
        grads  (dict): ∂L/∂W (各パラメータの勾配)
        Momentum 更新則:
            v_t = α⋅v_{t−1} − η⋅∂L/∂W
            W   = W + v_t
        """
        # 初回呼び出し時: v_0 = 0 で初期化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 各パラメータ W の更新
        for key in params.keys():
            # v_t = α⋅v_{t−1} − η⋅(∂L/∂W)
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # W ← W + v_t
            params[key] += self.v[key]
