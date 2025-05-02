#include <stdio.h>
#include <math.h>
#include <string.h>

// 配列をヒープ化する関数
void heapify(int arr[], int n, int i) {
    int largest = i; // 現在のノードを最大と仮定
    int left = 2 * i + 1; // 左の子
    int right = 2 * i + 2; // 右の子

    // 左の子が存在し、かつ親より大きければ、最大を左の子に更新
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }

    // 右の子が存在し、かつ最大より大きければ、最大を右の子に更新
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }

    // 最大が親でない場合、親と子を交換し、再帰的にヒープ化
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;

        // 交換後の部分木もヒープ化する
        heapify(arr, n, largest);
    }
}

// スペースを指定回数出力する関数
void printSpaces(int count) {
    for (int i = 0; i < count; i++) {
        printf(" ");
    }
}

// 配列の中身を表示する関数
void printArray(int arr[], int n) {
    printf("Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 木構造の形でヒープを表示する関数（親子関係を整え、中央寄せ）
void printHeapAsTree(int arr[], int n) {
    int level = 0;
    int index = 0;
    int maxLevel = 0;

    while ((1 << maxLevel) - 1 < n) maxLevel++;

    while (index < n) {
        int nodesInLevel = 1 << level;
        int spaceBefore = (1 << (maxLevel - level)) - 1;
        int spaceBetween = (1 << (maxLevel - level + 1)) - 1;

        printSpaces(spaceBefore);

        for (int i = 0; i < nodesInLevel && index < n; i++) {
            printf("%2d", arr[index++]);
            printSpaces(spaceBetween);
        }
        printf("\n");

        if (index < n) {
            printSpaces(spaceBefore - 1);
            for (int i = 0; i < nodesInLevel && (index + i) < n; i++) {
                printf("/\\");
                printSpaces(spaceBetween - 2);
            }
            printf("\n");
        }
        level++;
    }
    printf("\n");
}

// ヒープソート本体
void heapSort(int arr[], int n) {
    // (1) 配列全体を最大ヒープに構築する（下から順にヒープ化）
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
        printf("Heap after heapifying index %d:\n", i);
        printArray(arr, n);
        printHeapAsTree(arr, n);
    }

    // (2) ヒープから最大値（根）を取り出して、末尾と交換→サイズを減らして繰り返し
    for (int i = n - 1; i > 0; i--) {
        // 根（最大）と末尾を交換
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        printf("Heap after swapping root with index %d:\n", i);
        printArray(arr, n);
        printHeapAsTree(arr, n);

        // 縮小したヒープに対して再ヒープ化
        heapify(arr, i, 0);
        printf("Heap after re-heapifying size %d:\n", i);
        printArray(arr, i);
        printHeapAsTree(arr, i);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array:\n");
    printArray(arr, n);
    printHeapAsTree(arr, n);

    heapSort(arr, n);

    printf("\nSorted array (Heap Sort result):\n");
    printArray(arr, n);
    printHeapAsTree(arr, n);

    return 0;
}
