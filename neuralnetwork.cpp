#include "neuralnetwork.hpp"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

NeuralNetwork::NeuralNetwork(int input, int hidden, int output, double lr){
    inputSize = input;
    hiddenSize = hidden;
    outputSize = output;
    learningRate = lr; 
    
    weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
    weightsHiddenOutput.resize(hiddenSize, vector<double>(outputSize));

    randomGenerator = mt19937(randomDevice());
    distribution = normal_distribution<double>(0.0, 1.0);
    fillWeights(weightsInputHidden, inputSize, hiddenSize);
    fillWeights(weightsHiddenOutput, hiddenSize, outputSize);

    biasHidden.assign(hiddenSize, 0.0);
    biasOutput.assign(outputSize, 0.0);
}

NeuralNetwork::~NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(const NeuralNetwork & other) :
    inputSize(other.inputSize),
    hiddenSize(other.hiddenSize),
    outputSize(other.outputSize),
    learningRate(other.learningRate),
    weightsInputHidden(other.weightsInputHidden),
    weightsHiddenOutput(other.weightsHiddenOutput),
    biasHidden(other.biasHidden),
    biasOutput(other.biasOutput),
    randomGenerator(random_device()()),
    distribution(other.distribution)
{}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork & other) {
    if (this != &other) {
        inputSize = other.inputSize;
        hiddenSize = other.hiddenSize;
        outputSize = other.outputSize;
        learningRate = other.learningRate;
        weightsInputHidden = other.weightsInputHidden;
        weightsHiddenOutput = other.weightsHiddenOutput;
        biasHidden = other.biasHidden;
        biasOutput = other.biasOutput;
        randomGenerator = mt19937(random_device()());
        distribution = other.distribution;
    }
    return *this;
}


vector<double> NeuralNetwork::feedforward(const vector<double>& input){
    hiddenInput = vectorOperation(dotProduct(input, weightsInputHidden), biasHidden, add);
    hiddenOutput = activationFunction(hiddenInput, sigmoid);

    outputInput = vectorOperation(dotProduct(hiddenOutput, weightsHiddenOutput), biasOutput, add);
    predictedOutput = activationFunction(outputInput, sigmoid);
    return predictedOutput; 
}

void NeuralNetwork::backward(const vector<double>& input, const vector<double>& predicted, const vector<double>& actual, double lr){
    
    vector<double> outputError = vectorOperation(actual, predicted, subtract);
    vector<double> outputDelta = vectorOperation(outputError, activationFunction(predicted, sigmoidDerivative), multiply);
    
    std::vector<double> hiddenError(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            hiddenError[i] += outputDelta[j] * weightsHiddenOutput[i][j];
        }
    }

    vector<double> hiddenDelta = vectorOperation(hiddenError, activationFunction(hiddenOutput, sigmoidDerivative), multiply);
    
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightsHiddenOutput[i][j] += lr * hiddenOutput[i] * outputDelta[j];
        }
    }

    for (int j = 0; j < outputSize; ++j) {
        biasOutput[j] += lr * outputDelta[j];
    }

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] += lr * input[i] * hiddenDelta[j];
        }
    }

    for (int j = 0; j < hiddenSize; ++j) {
        biasHidden[j] += lr * hiddenDelta[j];
    }
}

double NeuralNetwork::mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / y_true.size();
}


void NeuralNetwork::train(const vector<vector<double>> & all_inputs, const vector<vector<double>> & all_actual_outputs, int epochs){
     for(int i = 0; i < epochs; i++){
        double total_loss = 0;
        for (size_t j = 0; j < all_inputs.size(); ++j) {
            const auto& input = all_inputs[j];
            const auto& actualOutput = all_actual_outputs[j];
            
            vector<double> predicted = feedforward(input);
            backward(input, predicted, actualOutput, this->learningRate);
            total_loss += mean_squared_error(actualOutput, predicted);
        }
        if (i % 1000 == 0) {
            std::cout << "Epoch " << i << ", Loss: " << total_loss / all_inputs.size() << "\n";
        }
    }
}
   

double NeuralNetwork::add(double a, double b){
    return a + b;
}

double NeuralNetwork::multiply(double a, double b){
    return a * b;
}

double NeuralNetwork::subtract(double a, double b){
    return a - b;
}

double NeuralNetwork::sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x){
    return x * (1.0 - x);
}

vector<double> NeuralNetwork::vectorOperation(const vector<double> & a, const vector<double> & b, function<double(double, double)> func){
    vector<double> result(a.size(), 0.0);
    for(size_t i = 0; i < a.size();i++){
        result[i] = func(a[i], b[i]);
    }
    return result;
}

vector<double> NeuralNetwork::vectorOperation(const vector<double> & a, const vector<vector<double>> & b, function<double(double, double)> func){
    vector<double> result(a.size(), 0.0);
    for(size_t i = 0; i < a.size();i++){
        for(size_t j = 0; j < b.size(); j++ ){
             result[i] += func(a[j], b[i][j]);
        }
    }
    return result;
}

vector<double> NeuralNetwork::activationFunction(const vector<double> & a, function<double(double)> func){
    vector<double> result(a.size(), 0.0);
    for(size_t i = 0; i < a.size();i++){
        result[i] = func(a[i]);
    }
    return result;
}

vector<double> NeuralNetwork::dotProduct(const vector<double> & input, const vector<vector<double>> & weights){
    size_t rows = weights.size();
    size_t cols = weights[0].size();
    vector<double> result(cols, 0.0);
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            result[j] += input[i] * weights[i][j];
        }
    }
    return result;
}

void NeuralNetwork::fillWeights(vector<vector<double>> & weights, int size1, int size2){
    weights = vector<vector<double>>(size1, vector<double>(size2, 0.0));
    for(int i = 0; i < size1; i++){
        for (int j = 0; j < size2; j++){
            weights[i][j] = distribution(randomGenerator);
        }
    }
}

int main(){
    //XOR 
    vector<vector<double>> X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> y_train = {{0}, {1}, {1}, {0}};

    NeuralNetwork nn(2, 500, 1, 0.1);

    cout << "Training the neural network for the XOR problem..." << endl;
    nn.train(X_train, y_train, 10000);

    cout << "\nTesting the neural network after training:" << endl;
    for(const auto& test_input : X_train) {
        vector<double> prediction = nn.feedforward(test_input);
        cout << "Input: [" << test_input[0] << ", " << test_input[1] << "] -> Prediction: " << prediction[0] << endl;
    }
   
    return 0;
}