#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

class NeuralNetwork {
private:
    int numInput;
    int numHidden;
    int numOutput;
    double learningRate;
    std::vector<std::vector<float>> inputToHiddenWeight;
    std::vector<std::vector<float>> hiddenToOutputWeight;
    std::vector<float> inputLayer;
    std::vector<float> hiddenLayer;
    std::vector<float> outputLayer;
    std::vector<float> hiddenErrors; // Erros na camada oculta
    std::vector<float> outputErrors; // Erros na camada de saída

public:
    // construtor
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double lr)
        : numInput(inputNodes), numHidden(hiddenNodes), numOutput(outputNodes), learningRate(lr) {
        // Inicializa os pesos aleatoriamente
        std::srand(std::time(nullptr)); // Inicializa a semente do gerador de números aleatórios

        // Pesos da camada de entrada para a camada oculta
        inputToHiddenWeight.resize(numInput, std::vector<float>(numHidden));
        for (int i = 0; i < numInput; ++i) {
            for (int j = 0; j < numHidden; ++j) {
                inputToHiddenWeight[i][j] = ((float)std::rand() / RAND_MAX) - 0.5;
            }
        }

        // Pesos da camada oculta para a camada de saída
        hiddenToOutputWeight.resize(numHidden, std::vector<float>(numOutput));
        for (int i = 0; i < numHidden; ++i) {
            for (int j = 0; j < numOutput; ++j) {
                hiddenToOutputWeight[i][j] = ((float)std::rand() / RAND_MAX) - 0.5;
            }
        }

        inputLayer.resize(numInput);
        hiddenLayer.resize(numHidden);
        outputLayer.resize(numOutput);
        hiddenErrors.resize(numHidden);
        outputErrors.resize(numOutput);
    }

    void propagateForward() {
        // Calcular as saídas da camada oculta
        for (int j = 0; j < numHidden; ++j) {
            float sum = 0;
            for (int i = 0; i < numInput; ++i) {
                sum += inputLayer[i] * inputToHiddenWeight[i][j];
            }
            hiddenLayer[j] = sigmoid(sum); // Aplicar função de ativação
        }

        // Calcular as saídas da camada de saída
        for (int j = 0; j < numOutput; ++j) {
            float sum = 0;
            for (int i = 0; i < numHidden; ++i) {
                sum += hiddenLayer[i] * hiddenToOutputWeight[i][j];
            }
            outputLayer[j] = sigmoid(sum); // Aplicar função de ativação
        }
    }

    void calcErrors(const std::vector<float>& expectedOutput) {
        for (int j = 0; j < numOutput; ++j) {
            outputErrors[j] = expectedOutput[j] - outputLayer[j];
        }
    }

    void propagateBackward(const std::vector<float>& expectedOutput) {

        calcErrors(expectedOutput);

        // Calcular os gradientes de erro na camada de saída
        std::vector<float> outputGradients(numOutput);
        for (int j = 0; j < numOutput; ++j) {
            float error = outputErrors[j];
            outputGradients[j] = error * sigmoidDerivative(outputLayer[j]); // Gradiente do erro
        }

        // Propagar os erros para a camada oculta
        std::vector<float> hiddenGradients(numHidden);
        for (int j = 0; j < numHidden; ++j) {
            float sum = 0;
            for (int k = 0; k < numOutput; ++k) {
                sum += outputGradients[k] * hiddenToOutputWeight[j][k];
            }
            hiddenGradients[j] = sum * sigmoidDerivative(hiddenLayer[j]);
        }

        // Atualizar os pesos da camada de saída
        for (int i = 0; i < numHidden; ++i) {
            for (int j = 0; j < numOutput; ++j) {
                hiddenToOutputWeight[i][j] += learningRate * outputGradients[j] * hiddenLayer[i];
            }
        }

        // Atualizar os pesos da camada oculta
        for (int i = 0; i < numInput; ++i) {
            for (int j = 0; j < numHidden; ++j) {
                inputToHiddenWeight[i][j] += learningRate * hiddenGradients[j] * inputLayer[i];
            }
        }
    }

    void train(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                inputLayer = input[i];

                propagateForward();

                calcErrors(target[i]);

                propagateBackward(target[i]);
            }

            float exampleLoss = 0.0;
            for (int j = 0; j < numOutput; ++j) {
                exampleLoss += 0.5 * std::pow(outputErrors[j], 2);
            }

            totalLoss += exampleLoss;

            float meanLoss = totalLoss / input.size();

            if ((epoch + 1) % 100 == 0 || epoch == epochs - 1) {
                std::cout << "epoch " << epoch + 1 << "/" << epochs << ", rms loss: " << meanLoss << std::endl;
            }
        }
    }

    std::vector<float> predict(const std::vector<float>& input) {
        inputLayer = input;
        propagateForward();
        return outputLayer;
    }
};
