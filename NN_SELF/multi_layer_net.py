import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(current_path, "..", ".."))  # この行をコメントアウトして、NN_SELF内で完結させる

import numpy as np  # numpyライブラリをnpというエイリアスでインポート（数値計算用）
import matplotlib.pyplot as plt  # グラフ描画用ライブラリmatplotlibのpyplotモジュールをpltとしてインポート
from functions import sigmoid, softmax, relu, cross_entropy_error  # 必要な活性化関数と損失関数をfunctionsモジュールからインポート
from optimizer import Adam  # 同一パッケージ内のoptimizerモジュールから最適化手法をインポート
import time  # 実行時間計測用
from collections import OrderedDict  # OrderedDictをインポート
from layers import Affine, Sigmoid, Relu, SoftmaxWithLoss  # 各レイヤの実装をインポート
from mnist import load_mnist


class MultiLayerNet:  # 多層パーセプトロン（MLP）を実現するクラスの定義
    def __init__(self, input_size=784, hidden_dims=[50], output_size=10,
                 hidden_activation=sigmoid, weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_dims
        self.hidden_layer_num = len(hidden_dims)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self.__init_weight(weight_init_std)
        
        # 各レイヤーオブジェクトをOrderedDictで管理する（順伝播・逆伝播のため）
        activation_layer = {sigmoid: Sigmoid, relu: Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[hidden_activation]()
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        
        # 出力層のsoftmaxと損失関数のレイヤーを定義
        self.last_layer = SoftmaxWithLoss()
        
    def __init_weight(self, weight_init_std):
        # 全層のユニット数リスト（入力層、隠れ層、出力層）
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(all_size_list)):
            if weight_init_std == "relu":
                scale = np.sqrt(2.0 / all_size_list[i-1])
            elif weight_init_std == "sigmoid":
                scale = np.sqrt(1.0 / all_size_list[i-1])
            else:
                scale = weight_init_std
            self.params["W" + str(i)] = np.random.randn(all_size_list[i-1], all_size_list[i]) * scale
            self.params["b" + str(i)] = np.zeros(all_size_list[i])
    
    def predict(self, x):
        # 各レイヤーのforwardを順次適用
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):  # 損失関数の値を計算するメソッド
        y = self.predict(x)  # 予測結果を計算
        return self.last_layer.forward(y, t)  # 最終層のforwardを使用して損失を計算
    
    def accuracy(self, x, t):  # 予測精度（正解率）を計算するメソッド
        y = self.predict(x) # 予測結果を計算
        y_pred = np.argmax(y, axis=1) # 予測結果から最大確率のインデックスを取得（予測ラベル）
        if t.ndim != 1: # tが one-hot 表現の場合のみ次元を変換
            t_label = np.argmax(t, axis=1)
        else:
            t_label = t
        return np.sum(y_pred == t_label) / float(x.shape[0])  # 正解ラベルとの一致率を計算して返す
    
    def gradient(self, x, t):  # 誤差逆伝播法により各パラメータの勾配を計算するメソッド
        # forward: 損失計算で順伝播を実施
        self.loss(x, t)

        # backward: 最終層から各レイヤーのbackwardを順に呼び出して逆伝播を実施
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 各Affineレイヤーの勾配を取得（正則化項も加味）
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
