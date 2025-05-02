#include <iostream>
using namespace std;

int main(){
    int n = 3;
    for(int k = 1; k <= n; k++){
        for(int l = 1; l <= n; l++){ // Changed < to <= for the inner loop
            cout << "k: " << k << ", l: " << l << endl;
        }
    }

    return 0;
}