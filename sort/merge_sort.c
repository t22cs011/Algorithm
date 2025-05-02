#include <stdio.h>

// 配列の状態を表示する関数
void printArray(int arr[], int left, int right) {
    for (int i = left; i <= right; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 2つの部分配列をマージする関数
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1; // 左部分の長さ
    int n2 = right - mid;    // 右部分の長さ

    int L[n1], R[n2]; // 左右の部分配列を用意

    // 左右の部分配列にデータをコピー
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    printf("マージ前: 左 = ");
    printArray(L, 0, n1 - 1);
    printf("マージ前: 右 = ");
    printArray(R, 0, n2 - 1);

    // マージ処理（小さい方を順にarrに格納）
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // 左部分に残りがあればコピー
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // 右部分に残りがあればコピー
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    printf("マージ後: ");
    printArray(arr, left, right);
    printf("\n");
}

// マージソート本体（再帰的に分割していく）
void mergeSort(int arr[], int left, int right) {
    if (left < right) { // 分割可能なら分割(left == rightとなれば配列の長さが1になったということ。そうなるまで分割する)
        int mid = left + (right - left) / 2;

        printf("分割: ");
        printArray(arr, left, right);

        // 左半分を再帰的にマージソート
        mergeSort(arr, left, mid);

        // 右半分を再帰的にマージソート
        mergeSort(arr, mid + 1, right);

        // 分割が終わったらマージ
        merge(arr, left, mid, right);
    }
}

// メイン関数
int main() {
    int arr[] = {12, 11, 13, 5, 6, 7}; // ソートする配列
    int arr_size = sizeof(arr) / sizeof(arr[0]);

    printf("元の配列: \n");
    printArray(arr, 0, arr_size - 1);
    printf("\n");

    // マージソート実行
    mergeSort(arr, 0, arr_size - 1);

    printf("ソート後の配列: \n");
    printArray(arr, 0, arr_size - 1);
    return 0;
}
