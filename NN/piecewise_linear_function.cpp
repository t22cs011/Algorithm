#include <iostream>
#include <vector>
using namespace std;

int piecewise_linear_function(vector<int> x, vector<int> w, double h){
    int y = 0; // 入力総和を初期化

    cout << "Calculation steps:" << endl;
    for(size_t i = 0; i < x.size(); i++){
        cout << "x[" << i << "] * w[" << i << "] = " << x[i] << " * " << w[i] << " = " << x[i] * w[i] << endl;
        y += x[i] * w[i];
    }

    cout << "Total potential (y): " << y << endl;
    cout << "Threshold (h): " << h << endl;

    if(y >= h) {
        cout << "y >= h, returning 1" << endl;
        return 1;
    } else {
        cout << "y < h, returning 0" << endl;
        return 0;
    }
}

int main(){
    vector<int> x = {1, 0, 1}; // 入力
    vector<int> w = {2, -1, 3}; // シナプス重み
    double h = 2.5; // 閾値

    int result = step_function(x, w, h);
    cout << "The output of the neuron is: " << result << endl;

    return 0;
}