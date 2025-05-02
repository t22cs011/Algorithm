class SGD:
    def __inti__(self, lr = 0.01):
        self.lr = lr #lr: learning rate(学習係数)

    def update(self, params, grads): #params, grads: ディクショナリ変数
        for key in params.keys(): #params["W1"], grads["W1"]のようにそれぞれ重みとパラメータが格納されている
            params[key] -= self.lr * grads[key]

    
    network = TwoLayerNet(...)
    optimizer = SGD() #パラメータの更新(最適化)を行うもの. ここをMomentumにすればSGDがMomentumに切り替わる
    #最適化を行うクラスを分離して実装することで, 昨日のモジュール化が容易になる

    for i in range(10000):
        ...
        x_batch, t_batch = get_mini_batch(...) #ミニバッチ
        grads = network.gradient(x_batch, t_batch)
        params = network.params
        optimizer.update(params, grads)
        ...