#include <stdio.h>

void bubbleSort(int arr[], int n) {
    int i, j, temp;

    // ステップ1: 配列の先頭から末尾-1までを繰り返し走査する
    for (i = 0; i < n - 1; i++) {
        // ステップ2: 各パスで隣接要素を比較して並べ替える
        for (j = 0; j < n - 1 - i; j++) {
            // ステップ3: 現在の要素と次の要素を比較
            if (arr[j] > arr[j + 1]) {
                // ステップ4: 順番が逆なら交換する
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
        // ステップ5: 1回のパス終了時点で一番大きい要素が末尾に確定
        printf("After pass %d: ", i + 1);
        for (int k = 0; k < n; k++) {
            printf("%d ", arr[k]);
        }
        printf("\n");
    }
}

int main() {
    int arr[] = {5, 3, 8, 4, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");

    bubbleSort(arr, n);

    printf("\nSorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}