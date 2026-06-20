

#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>


std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& input) {
    if (input.empty()) {
        return {};
    }
    
    int inputRows = input.size();
    int inputCols = input[0].size();
    std::vector<std::vector<double>> output(
        inputCols,
        std::vector<double>(inputRows)
    );

    for (int i = 0; i < inputRows; ++i) {
        for (int j = 0; j < inputCols; j++) {
            output[j][i] = input[i][j];
        }
    }
    return output;
}


std::vector<std::vector<double>> matMul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("Matrices cannot be empty");
    }

    int rowsA = A.size();
    int colsA = A[0].size();

    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }

    std::vector<std::vector<double>> output(
        rowsA,
        std::vector<double>(colsB, 0.0)
    );

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double multiplication = 0.0;

            for (int k = 0; k < colsA; ++k) {
                multiplication += A[i][k] * B[k][j];
            }

            output[i][j] = multiplication;
        }
    }

    return output;
}




class ReLU {

    std::vector<double>& input_;

public:
    std::vector<double> forward(const std::vector<double>& x) {
        std::vector<double> output = x; /*copy and then max between 0 and value*/
        for (double& val : output) {
            val = std::max(val, 0.0);
        }
        input_ = x;
        return output;
    }

    std::vector<double> backward(const std::vector<double>& z) {
        /*
        Compute influence and return value
        */
    }

};




class Layer {

    int inputSize_;
    int outputSize_;
    std::vector<std::vector<double>> W;
    std::vector<double> b;

    void initializeWandB() {
        /* Assign random values between -0.5 and 0.5 to W matrix and b vector */
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        W.resize(outputSize_, std::vector<double>(inputSize_));
        b.resize(outputSize_);

        for (int i = 0; i < outputSize_; ++i) {
            for (int j = 0; j < inputSize_; ++j) {
                W[i][j] = dist(gen);
            }

            b[i] = dist(gen);
        }
    }


public:

    Layer(int inputSize, int outputSize) 
        : inputSize_{inputSize} , outputSize_{outputSize} {
            initializeWandB();
        }


    std::vector<double> forward(std::vector<double>& input) {
        /*
        z = Wx + b
        a = activation_function(z)
        We have to store values for backtrack
        */
       std::vector<double> a;
       return a;
    }

    std::vector<double> backward(std::vector<double>& output) {
        /*
        Compute derivatives and store them to update after each step
        Also compute how much input contributes to pass it to the prev layer
        So here we compute dW , db and dx , and we return dx to the prev layer
        */
        std::vector<double> inputProportion;
        return inputProportion;
    }

};