#include <stdio.h>

void quicksort(int arr[], int left, int right) {
    if (left >= right) return;

    // ステップ① pivot (P) を決める：右端の値をピボットとする
    int P = arr[right];

    // ステップ② 左右マーカー (L, R) を初期化
    int L = left;
    int R = right - 1;

    // ステップ③ 左マーカー (L) を右に進める
    // ステップ④ 右マーカー (R) を左に進める
    while (1) {
        while (L <= R && arr[L] < P) L++;  // LはPより小さい間、右に進む（止まらない）
        while (R >= L && arr[R] > P) R--;  // RはPより大きい間、左に進む（交差で停止）

        if (L >= R) break;  // ステップ⑤ LとRが交差したらループを抜ける

        // ステップ⑤ (続き) LとRが交差する前なら値を交換
        int temp = arr[L];
        arr[L] = arr[R];
        arr[R] = temp;

        L++;
        R--;
    }

    // ステップ⑥ pivot (P) をLの位置に移動
    int temp = arr[L];
    arr[L] = arr[right];
    arr[right] = temp;

    // ステップ⑦ 左右に再帰的にソート
    quicksort(arr, left, L - 1);
    quicksort(arr, L + 1, right);
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {4, 2, 7, 1, 9, 5, 3, 8, 6};
    int n = sizeof(arr) / sizeof(arr[0]);
    printArray(arr, n);
    quicksort(arr, 0, n - 1);
    printArray(arr, n);

    return 0;
}
