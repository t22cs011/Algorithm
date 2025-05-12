#!/usr/bin/env python3
# coding: utf-8

import pandas as pd

def main():
    results = []
    for i in range(1, 15):
        file_path = f"NN_SELF/experiment_A{i}_log.csv"
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(file_path)
            
            # Epochが200の行を抽出（存在しない場合は最終行を使用）
            final_row = df[df["Epoch"] == 200]
            if final_row.empty:
                final_row = df.iloc[[-1]]
            
            # 最終値の取得
            final_train = final_row["Training Accuracy"].values[0]
            final_test = final_row["Test Accuracy"].values[0]
            
            # CSV全体の平均精度を計算
            avg_train = df["Training Accuracy"].mean()
            avg_test = df["Test Accuracy"].mean()
            
            # グラフにあるexecution-timeの値　（CSVに "execution-time" 列がある場合）
            if "execution-time" in df.columns:
                exec_time = final_row["execution-time"].values[0]
            else:
                exec_time = ""
            
            results.append({
                "実験パターン": f"A{i}",
                "最終訓練精度": final_train,
                "最終テスト精度": final_test,
                "平均訓練精度": avg_train,
                "平均テスト精度": avg_test,
                "実行時間": exec_time
            })
        except Exception as e:
            print(f"{file_path} の読み込みに失敗しました: {e}")
    
    # 結果をDataFrameにまとめる
    summary = pd.DataFrame(results)
    
    # 各指標の平均値を計算する（「実行時間」は数値の場合のみ算出）
    avg_row = {
        "実験パターン": "平均",
        "最終訓練精度": summary["最終訓練精度"].mean(),
        "最終テスト精度": summary["最終テスト精度"].mean(),
        "平均訓練精度": summary["平均訓練精度"].mean(),
        "平均テスト精度": summary["平均テスト精度"].mean(),
    }
    try:
        summary["実行時間_numeric"] = pd.to_numeric(summary["実行時間"])
        avg_row["実行時間"] = summary["実行時間_numeric"].mean()
    except Exception:
        avg_row["実行時間"] = ""
        
    avg_df = pd.DataFrame([avg_row])
    final_summary = pd.concat([summary, avg_df], ignore_index=True)
    
    # CSV形式で保存
    final_summary.to_csv("accuracy_summary.csv", index=False)
    print("精度の表を 'accuracy_summary.csv' に保存しました。")

if __name__ == "__main__":
    main()