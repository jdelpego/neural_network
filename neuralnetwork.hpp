#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <random>
#include <vector>
using namespace std;


class NeuralNetwork{
public:
    int inputSize;
    int hiddenSize;
    int outputSize;

    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;

    vector<double> biasHidden;
    vector<double> biasOutput;

    random_device randomDevice;
    mt19937 randomGenerator;
    normal_distribution<double> distribution;

    void fillWeights(vector<vector<double>> & weights, int size1, int size2);
    double sigmoid(double x);
    double sigmoid_derivative(double x);

    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(const NeuralNetwork & other);
    NeuralNetwork& operator=(const NeuralNetwork & other);
};




#endif