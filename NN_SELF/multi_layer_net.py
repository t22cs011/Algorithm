import numpy as np  # numpyライブラリをnpというエイリアスでインポート（数値計算用）
import matplotlib.pyplot as plt  # グラフ描画用ライブラリmatplotlibのpyplotモジュールをpltとしてインポート
from functions import sigmoid, softmax, relu, cross_entropy_error  # 必要な活性化関数と損失関数をfunctionsモジュールからインポート
from optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam  # 同一パッケージ内のoptimizerモジュールから最適化手法をインポート
import time  # 実行時間計測用

# 以下の設定で日本語フォントをAppleGothicに変更（macOSの場合）
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
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
                 hidden_activation=sigmoid):  # コンストラクタ: 入力サイズ、中間層の構造、出力サイズを指定
        self.layers = [input_size] + hidden_dims + [output_size]  # ネットワークの各層のサイズをリストで定義（入力層→中間層→出力層）
        self.hidden_activation = hidden_activation
        self.output_activation = softmax  # 出力層は常にsoftmaxで固定
        self.num_layers = len(self.layers) - 1  # 重みを持つレイヤーの数（中間層と出力層の合計）を計算
        self.params = {}  # パラメータ（重みとバイアス）を保存する辞書を初期化
        for i in range(self.num_layers):  # 各レイヤーに対してパラメータを初期化
            self.params['W' + str(i+1)] = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])  # He初期化を用いて重みをランダムに設定
            self.params['b' + str(i+1)] = np.zeros(self.layers[i+1])  # バイアスはゼロで初期化
    
    def predict(self, x):  # 順伝播により予測結果を計算するメソッド
        out = x  # 入力データを出力の初期値に設定
        for i in range(1, self.num_layers):  # 中間層までループ
            W = self.params['W' + str(i)]  # i層目の重みを取得
            b = self.params['b' + str(i)]  # i層目のバイアスを取得
            out = self.hidden_activation(np.dot(out, W) + b)  # 線形変換にhidden_activationを適用
        W = self.params['W' + str(self.num_layers)]  # 出力層の重みを取得
        b = self.params['b' + str(self.num_layers)]  # 出力層のバイアスを取得
        out = self.output_activation(np.dot(out, W) + b)  # 出力層で線形変換にoutput_activationを適用し確率分布を計算
        return out  # 予測結果（確率分布）を返す
    
    def loss(self, x, t):  # 損失関数の値を計算するメソッド
        y = self.predict(x)  # 予測結果を計算
        return cross_entropy_error(y, t)  # 交差エントロピー誤差を計算して返す
    
    def accuracy(self, x, t):  # 予測精度（正解率）を計算するメソッド
        y = self.predict(x)  # 予測結果を計算
        y_pred = np.argmax(y, axis=1)  # 予測結果から最大確率のインデックスを取得（予測ラベル）
        t_label = np.argmax(t, axis=1)  # 正解ラベル（one-hotの場合）のインデックスを取得
        return np.sum(y_pred == t_label) / float(x.shape[0])  # 正解ラベルとの一致率を計算して返す
    
    def gradient(self, x, t):  # 誤差逆伝播法により各パラメータの勾配を計算するメソッド
        grads = {}  # 勾配を保存する辞書を初期化
        batch_num = x.shape[0]  # バッチサイズを取得
        activations = [x]  # 各層の出力（活性化値）を保存するリスト。初期値は入力データ
        pre_activations = []  # 各層の線形変換後の値（活性化前）を保存するリスト
        for i in range(1, self.num_layers + 1):  # 各層に対して順伝播を実施
            W = self.params['W' + str(i)]  # i層目の重みを取得
            b = self.params['b' + str(i)]  # i層目のバイアスを取得
            a = np.dot(activations[i-1], W) + b  # 線形変換を計算
            pre_activations.append(a)  # 線形変換結果を保存
            if i == self.num_layers:  # 最終層の場合
                z = self.output_activation(a)  # output_activationを適用
            else:  # 中間層の場合
                z = self.hidden_activation(a)  # hidden_activationを適用
            activations.append(z)  # 活性化結果をリストに追加
        delta = (activations[-1] - t) / batch_num  # 出力層の誤差（交差エントロピーとソフトマックスの組み合わせによる微分）を計算
        grads['W' + str(self.num_layers)] = np.dot(activations[-2].T, delta)  # 出力層の重みの勾配を計算
        grads['b' + str(self.num_layers)] = np.sum(delta, axis=0)  # 出力層のバイアスの勾配を計算
        for i in range(self.num_layers - 1, 0, -1):  # 逆伝播を中間層に向かって実施
            W_next = self.params['W' + str(i+1)]  # 次の層の重みを取得
            if self.hidden_activation == sigmoid:
                derivative = activations[i] * (1 - activations[i])
            else:
                derivative = (pre_activations[i-1] > 0).astype(float)
            delta = np.dot(delta, W_next.T) * derivative  # 誤差を逆伝播し、活性化関数の微分で重み付け
            grads['W' + str(i)] = np.dot(activations[i-1].T, delta)  # i層目の重みの勾配を計算
            grads['b' + str(i)] = np.sum(delta, axis=0)  # i層目のバイアスの勾配を計算
        return grads  # すべての層の勾配を返す


if __name__ == '__main__':  # 自動実験モード
    experiments = [
        # 🟢 汎化できている（ベースライン）
        {"hidden_dims": [64], "epochs": 50, "batch_size": 128, "lr": 0.01},
        {"hidden_dims": [128, 128], "epochs": 50, "batch_size": 128, "lr": 0.01},
        {"hidden_dims": [256, 256, 256], "epochs": 100, "batch_size": 128, "lr": 0.005},

        # 🟡 軽度〜中度の過学習
        {"hidden_dims": [512, 512, 512, 512], "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [1024, 1024], "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [1024]*6, "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [64]*20, "epochs": 100, "batch_size": 128, "lr": 0.01},
        {"hidden_dims": [8]*12, "epochs": 100, "batch_size": 128, "lr": 0.01},

        # 🔴 重度の過学習
        {"hidden_dims": [1024]*10, "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [2048, 1024, 512, 256], "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [512]*30, "epochs": 100, "batch_size": 128, "lr": 0.001},
        {"hidden_dims": [512]*6, "epochs": 300, "batch_size": 128, "lr": 0.0001},

        # ⚫ データ不足による過学習
        {"hidden_dims": [512, 512], "epochs": 100, "batch_size": 128, "lr": 0.001, "train_ratio": 0.25},
        {"hidden_dims": [1024, 1024], "epochs": 100, "batch_size": 128, "lr": 0.001, "train_ratio": 0.10},
        {"hidden_dims": [1024, 1024, 512], "epochs": 100, "batch_size": 128, "lr": 0.001, "train_ratio": 0.017}
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

    print("✅ All experiments completed")
