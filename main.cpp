

/*
Binary classifier in c++ from scratch
For the 1D vectors I am using 2D vectors, but with 1 dimension fixed to 1,
this way I can reuse the matrix multiplication function I created,
these "1D" matrixes will be (output_dim, 1)

log prob: y * log(y_pred) + (1 - y)*log(1 - y_pred)

- dloss/dW = dloss/dy_pred * dy_pred/dz * dz/dW 
- dloss/db = dloss/dy_pred * dy_pred/dz * dz/db
- dloss/dx = dloss/dy_pred * dy_pred/dz * dz/dx

z = Wx + b
- dz/dW = x
- dz/db = 1
- dz/dx = W

y_pred = activation = relu / sigmoid
- SIGMOID dy_pred/dz = sigmoid(z)(1-sigmoid(z))
- RELU    dy_pred/dz = o if z < 0 else 1

dloss/dy_pred = y/y_pred - (1-y)/(1-y_pred)

*/



#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <algorithm>


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

    std::vector<std::vector<double>> input_;

public:
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) {
        input_ = x;

        std::vector<std::vector<double>> output = x; /*copy and then max between 0 and value*/
        for (std::vector<double>& val : output) {
            val[0] = std::max(val[0], 0.0);
        }

        return output;
    }

     std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) {
        std::vector<std::vector<double>> gradInput = gradOutput;

        for (int i = 0; i < gradInput.size(); ++i) {
            if (input_[i][0] > 0.0) {
                gradInput[i][0] *= 1.0;
            } else {
                gradInput[i][0] = 0.0;
            }
        }

        return gradInput;
    }

};




class Layer {

    int inputSize_;
    int outputSize_;
    std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activationFunction_;

    std::vector<std::vector<double>> W; /* (output_dim, input_dim) */
    std::vector<std::vector<double>> b; /* (output_dim, 1) */
    std::vector<std::vector<double>> x; /* (input_dim, 1) stored for backtrack */
    
    void initializeWandB() {
        /* Assign random values between -0.5 and 0.5 to W matrix and b vector */
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        W.resize(outputSize_, std::vector<double>(inputSize_)); /* (output_dim, input_dim) */
        b.resize(outputSize_, std::vector<double>(1)); /* (output_dim, 1) */

        for (int i = 0; i < outputSize_; ++i) {
            for (int j = 0; j < inputSize_; ++j) {
                W[i][j] = dist(gen);
            }
            b[i][0] = dist(gen);
        }
    }


public:

    Layer(int inputSize, int outputSize, std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> activationFunction) 
        : inputSize_{inputSize} , outputSize_{outputSize}, activationFunction_{activationFunction} {
            initializeWandB();
        }


    std::vector<std::vector<double>> forward(std::vector<std::vector<double>>& input) {
        /*
        z = Wx + b
        a = activation_function(z)
        We have to store the input value for backtrack
        */
        
        std::vector<std::vector<double>> a = matMul(W, input); /*(output_dim, 1)*/
        a = activationFunction_(a); /*(output_dim, 1)*/
        return a;
    }

    std::vector<std::vector<double>> backward(std::vector<std::vector<double>>& output) {
        /*
        Compute derivatives and store them to update after each step
        Also compute how much input contributes to pass it to the prev layer
        So here we compute dW , db and dx , and we return dx to the prev layer
        */
        std::vector<std::vector<double>> inputProportion;
        return inputProportion;
    }

};