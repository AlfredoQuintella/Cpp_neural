#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "NeuralNetwork.hpp"

std::vector<std::vector<float>> input; 
std::vector<std::vector<float>> target;

void readData(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    int index = 0;
    std::vector<float> tempInput;
    std::vector<float> tempTarget(3, 0.0);

    while (std::getline(iss, token, ',')) {
        if (index < 4) {
            tempInput.push_back(std::stof(token));
        } else {
            if(token == "Iris-setosa") tempTarget[0] = 1.0;
            else if(token == "Iris-versicolor") tempTarget[1] = 1.0;
            else if(token == "Iris-virginica") tempTarget[2] = 1.0;
        }
        index++;
    }
    input.push_back(tempInput);
    target.push_back(tempTarget);
}

void readDataFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        readData(line);
    }
    
    file.close();
}

int identificador(std::vector<float>& resultado){
    int temp;
    if(resultado[0] == resultado[1] || resultado[1] == resultado[2] || resultado[0] == resultado[2]) return -1;
    if(resultado[0] > resultado[1]){
        if(resultado[0] > resultado[2]) return 0;
        else return 2;
    }
    else{
        if(resultado[1] > resultado[2]) return 1;
        else return 2;
    }
}

int main() {
    const std::string filename = "iris.data";
    
    readDataFromFile(filename);

    int numInputNodes = 4;
    int numHiddenNodes = 20;
    int numOutputNodes = 3; 
    double learningRate = 0.1;
    int epochs = 10000;

    NeuralNetwork neuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes, learningRate);

    neuralNetwork.train(input, target, epochs);

    std::vector<std::vector<float>> esperado;
    std::vector<std::vector<float>> teste = {{6.8, 3.2, 5.9, 2.3}, {7.0, 3.2, 4.7, 1.4}, {5.9, 3.0, 5.1, 1.8}};

    for (int i = 0; i < teste.size(); i++){
        std::vector<float> resultado = neuralNetwork.predict(teste[i]);
        esperado.push_back(resultado);
        std::cout << "Amostra " << i+1 << ": ";
        for (float val : esperado[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        switch(identificador(resultado)){
            case 0: 
            std::cout << "Iris-setosa" << std::endl;
            break;

            case 1:
            std::cout << "Iris-versicolor" << std::endl;
            break;

            case 2:
            std::cout << "Iris-virginica" << std::endl;
            break;
        }
    }

    return 0;
}
