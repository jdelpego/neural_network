#include "neuralnetwork.hpp"
#include <iostream>
#include <cmath>
using namespace std;

NeuralNetwork::NeuralNetwork(){
    inputSize = 1;
    hiddenSize = 1;
    outputSize = 1;
    
    randomGenerator = mt19937(randomDevice());
    distribution = normal_distribution<double>(0.0, 1.0);
    fillWeights(weightsInputHidden, inputSize, hiddenSize);
    fillWeights(weightsHiddenOutput, hiddenSize, outputSize);

    vector<double> biasHidden(hiddenSize, 0.0);
    vector<double> biasOutput(outputSize, 0.0);

}

double NeuralNetwork::sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x){
    return x * (1.0-x);
}

void NeuralNetwork::fillWeights(vector<vector<double>> & weights, int size1, int size2){
    for(int i = 0; i < size1; i++){
        for (int j = 0; j < size2; j++){
            weights[i][j] = distribution(randomGenerator);
        }
    }
}




int main(){
    return 0;
}