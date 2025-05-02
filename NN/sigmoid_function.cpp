#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

double sigmoid_function(vector<int> x, vector<int> w, double epsilon){
    int u = 0;

    cout << "Calculation steps:" << endl;
    for(size_t i = 0; i < x.size(); i++){
        cout << "x[" << i << "] * w[" << i << "] = " << x[i] << " * " << w[i] << " = " << x[i] * w[i] << endl;
        u += x[i] * w[i];
    }

    cout << "Total input sum (u): " << u << endl;
    cout << "Epsilon (ε): " << epsilon << endl;

    // ε付きシグモイド関数
    double output = 1.0 / (1.0 + exp(-epsilon * u));
    cout << "Sigmoid output (1 / (1 + exp(-εu))): " << output << endl;

    return output;
}

int main(){
    vector<int> x = {1, 0, 1};
    vector<int> w = {2, -1, 3};
    double epsilon = 1.0; // εの値（お好みで調整）

    double result = sigmoid_function(x, w, epsilon);
    cout << "The output of the neuron is: " << result << endl;

    return 0;
}