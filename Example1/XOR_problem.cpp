#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main(){
    int numInputNodes = 3;
    int numHiddenNodes = 10;
    int numOutputNodes = 2;
    double learningRate = 0.1;
    int epochs = 100000;

    NeuralNetwork neuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes, learningRate);

    vector<vector<float>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> target = {{0}, {1}, {1}, {0}};

    neuralNetwork.train(input, target, epochs);

    for(size_t i = 0; i < input.size(); ++i) {
        vector<float> output = neuralNetwork.predict(input[i]);
        cout << "Input: " << input[i][0] << ", " << input[i][1]
             << " | Output: " << output[0] << endl;
    }

    return 0;
}
