import pickle
import numpy as np

# mnist.pklを読み込む
with open('mnist.pkl', 'rb') as f:
    dataset = pickle.load(f)

# データセットの構造を確認
print("データセットの型:", type(dataset))
print("\nデータセットの内容:")
for key, value in dataset.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key}: {type(value)}") 