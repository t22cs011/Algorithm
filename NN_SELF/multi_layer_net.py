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

# 以下の設定で日本語フォントを変更
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAexGothic'  # Linux用の日本語フォント
plt.rcParams['axes.unicode_minus'] = False


def load_mnist():  # MNISTデータセットを取得し前処理を行う関数
    """deep-learning-from-scratchのmnist.pyを使用してMNISTデータセットを取得し、前処理を行う関数
    normalize=True : ピクセル値を0～1に正規化
    flatten=True   : 画像を一次元配列に変換
    one_hot_label=True : ラベルをOne-hotエンコーディングする
    """
    from mnist import load_mnist as ds_load_mnist  # mnist.pyからload_mnist関数をインポート
    (X_train, T_train), (X_test, T_test) = ds_load_mnist(normalize=True, flatten=True, one_hot_label=True)  # MNISTデータセット取得、訓練・テストに分割
    return X_train, X_test, T_train, T_test


def one_hot(labels, num_classes=10):  # ラベルをone-hotエンコーディングする関数の定義
    one_hot_labels = np.zeros((labels.shape[0], num_classes))  # ラベル数×クラス数のゼロ行列を作成
    for idx, label in enumerate(labels):  # 各ラベルに対してループ
        one_hot_labels[idx, int(label)] = 1  # 該当するクラスの位置に1をセット
    return one_hot_labels  # one-hot表現されたラベルを返す


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
        y = self.predict(x)  # 予測結果を計算
        y_pred = np.argmax(y, axis=1)  # 予測結果から最大確率のインデックスを取得（予測ラベル）
        t_label = np.argmax(t, axis=1)  # 正解ラベル（one-hotの場合）のインデックスを取得
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


if __name__ == '__main__':  # 自動実験モード

    experiments = [
    # D1: 浅くて狭い → underfitting しやすい
    {"hidden_dims": [8], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D1"},

    # D2: 浅くて中くらい → ベースライン
    {"hidden_dims": [128], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D2"},

    # D3: 浅くて広い → 表現力向上 vs 過学習リスク
    {"hidden_dims": [512], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D3"},

    # D4: 少し深い＆狭い
    {"hidden_dims": [64, 64], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D4"},

    # D5: 少し深い＆中くらい → ベースライン + 軽度過学習の兆候
    {"hidden_dims": [128, 128], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D5"},

    # D6: 少し深い＆広い
    {"hidden_dims": [512, 512], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D6"},

    # D7: 深くて狭い → 学習の安定性
    {"hidden_dims": [64, 64, 64, 64], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D7"},

    # D8: 深くて中くらい → 過学習の度合い
    {"hidden_dims": [128, 128, 128, 128], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D8"},

    # D9: 深くて広い → 重度な過学習のリスク
    {"hidden_dims": [256, 256, 256, 256], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D9"},

    # D10: ボトルネック構造 → 情報圧縮の効果
    {"hidden_dims": [1024, 64, 1024], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D10"},

    # D11: ピラミッド（広→狭） → 階層的特徴抽出の過学習傾向
    {"hidden_dims": [512, 256, 128, 64], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D11"},

    # D12: 逆ピラミッド（狭→広） → 拡張による学習安定性
    {"hidden_dims": [64, 128, 256, 512], "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D12"},

    # D13: より深く狭い → 学習の難しさ
    {"hidden_dims": [256] * 8, "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D13"},

    # D14: より深く広い → 過学習と計算コスト
    {"hidden_dims": [1024] * 8, "epochs": 200, "batch_size": 128, "lr": 0.001, "experiment_id": "D14"},
    ]

    for idx, config in enumerate(experiments, 1):
        print(f"\n===== Running Experiment A{idx}: {config} =====")
        X_train, X_test, T_train, T_test = load_mnist()
        train_ratio = config.get("train_ratio", 1.0)
        if train_ratio < 1.0:
            N = int(len(X_train) * train_ratio)
            X_train, T_train = X_train[:N], T_train[:N]

        net = MultiLayerNet(
            input_size=784,
            hidden_dims=config["hidden_dims"],
            output_size=10,
            hidden_activation=relu
        )
        optimizer = Adam(lr=config["lr"])
        train_acc_list, test_acc_list = [], []
        iter_per_epoch = max(1, X_train.shape[0] // config["batch_size"])
        start_time = time.time()

        for epoch in range(config["epochs"]):
            for _ in range(iter_per_epoch):
                batch_mask = np.random.choice(X_train.shape[0], config["batch_size"])
                x_batch = X_train[batch_mask]
                t_batch = T_train[batch_mask]
                grads = net.gradient(x_batch, t_batch)
                optimizer.update(net.params, grads)
            train_acc = net.accuracy(X_train, T_train)
            test_acc = net.accuracy(X_test, T_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"Epoch {epoch+1}/{config['epochs']} - Train: {train_acc:.4f}, Test: {test_acc:.4f}")

        layer_str = '-'.join(str(n) for n in config["hidden_dims"])
        filename = (
            f"layers[{layer_str}]_ep{config['epochs']}_bs{config['batch_size']}"
            f"_lr{config['lr']}_Adam_mid-relu_out-softmax.png"
        )
        title = (
            f"Layers:{len(config['hidden_dims'])}({layer_str}) bs:{config['batch_size']} lr:{config['lr']} "
            f"opt:Adam hid_act:relu out_act:softmax"
        )
        elapsed = time.time() - start_time

        epochs_range = np.arange(1, config["epochs"]+1)
        plt.figure()
        plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
        plt.plot(epochs_range, test_acc_list, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.figtext(0.01, 0.02, f"Execution time: {elapsed:.2f} s", ha='left', va='bottom')
        plt.savefig(filename)
        print(f"✅ Saved: {filename} ({elapsed:.2f}秒)")
        plt.close()

        # CSVログの保存処理を追加（実験結果をCSV形式で記録）
        import csv
        csv_filename = f"experiment_A{idx}_log.csv"
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Training Accuracy", "Test Accuracy"])
            for epoch_num, (t_train, t_test) in enumerate(zip(train_acc_list, test_acc_list), start=1):
                writer.writerow([epoch_num, f"{t_train:.4f}", f"{t_test:.4f}"])
        print(f"✅ CSVログ保存: {csv_filename}")

    print("✅ All experiments completed")
