#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <random>
#include <vector>
#include <functional>
using namespace std;


class NeuralNetwork{
public:
    int inputSize;
    int hiddenSize;
    int outputSize;

    double learningRate;; 

    vector<double> hiddenInput;
    vector<double> hiddenOutput;

    vector<double> outputInput;
    vector<double> predictedOutput;

    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;

    vector<double> biasHidden;
    vector<double> biasOutput;

    random_device randomDevice;
    mt19937 randomGenerator;
    normal_distribution<double> distribution;

    void fillWeights(vector<vector<double>> & weights, int size1, int size2);
    vector<double> vectorOperation(const vector<double> & a, const vector<double> & b, function<double(double, double)> func);
    vector<double> vectorOperation(const vector<double> & a, const vector<vector<double>> & b, function<double(double, double)> func);

    vector<double> activationFunction(const vector<double> & a, function<double(double)> func);
    vector<double> dotProduct(const vector<double> & input, const vector<vector<double>> & weights);

    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    
    static double add(double a, double b);
    static double subtract(double a, double b);
    static double multiply(double a, double b);
    double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred);


    vector<double> feedforward(const vector<double>& input);
    void backward(const vector<double>& input, const vector<double>& predicted, const vector<double>& actual, double learning_rate);
    void train(const vector<vector<double>> &  input, const vector<vector<double>> & actualOutput, int epoch);

    NeuralNetwork(int input, int hidden, int output, double lr);
    ~NeuralNetwork();
    NeuralNetwork(const NeuralNetwork & other);
    NeuralNetwork& operator=(const NeuralNetwork & other);
};



#endif