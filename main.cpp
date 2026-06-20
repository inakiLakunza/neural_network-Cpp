

/*
Binary classifier in c++ from scratch
For the 1D vectors I am using 2D vectors, but with 1 dimension fixed to 1,
this way I can reuse the matrix multiplication function I created,
these "1D" matrixes will be (output_dim, 1)

log prob: y * log(y_pred) + (1 - y)*log(1 - y_pred)

- dloss/dW = dloss/dy_pred * dy_pred/dz * dz/dW 
- dloss/db = dloss/dy_pred * dy_pred/dz * dz/db
- dloss/dx = dloss/dy_pred * dy_pred/dz * dz/dx

dloss/dy_pred == dout

z = Wx + b
- dz/dW = x
- dz/db = 1
- dz/dx = W

y_pred = activation = relu / sigmoid
- SIGMOID dy_pred/dz = sigmoid(z)(1-sigmoid(z))
- RELU    dy_pred/dz = o if z < 0 else 1

dloss/dy_pred == dout = y/y_pred - (1-y)/(1-y_pred)

*/



#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>



constexpr double LR = 0.001;
constexpr double EPS = 1e-12;
constexpr int OUTPUTSIZE = 1;


std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& input) {
    if (input.empty()) {
        return {};
    }
    
    int inputRows = input.size();
    int inputCols = input[0].size();
    /* 
    vector constructors:
    std::vector<T> vector(size, value);
    std::vector<T> vector(size); -> initializes with 0s
    so below we are constructing a vector with inputCols elements
    and each is a std::vector<double> vector with inputRows elements
    and since we are using the constructor without values inside,
    this means that it is initialized with zeroes
    */
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
    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) {
        input_ = input;

        std::vector<std::vector<double>> output = input; /*copy and then max between 0 and value*/
        for (std::vector<double>& val : output) {
            val[0] = std::max(val[0], 0.0);
        }

        return output;
    }

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& gradOutput
    ) {
        std::vector<std::vector<double>> gradInput = gradOutput;

        for (std::size_t i = 0; i < gradInput.size(); ++i) {
            if (input_[i][0] > 0.0) {
                gradInput[i][0] *= 1.0;
            } else {
                gradInput[i][0] = 0.0;
            }
        }
        return gradInput;
    }

};


class Sigmoid {

    std::vector<std::vector<double>> output_;

public:
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) {
        std::vector<std::vector<double>> output = x;
        for (std::vector<double>& val : output) {
            val[0] = 1.0 / (1.0 + std::exp(-val[0]));
        }
        output_ = output;
        return output;
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) {
        std::vector<std::vector<double>> gradInput = gradOutput;
        for (std::size_t i = 0; i < gradInput.size(); ++i) {
            gradInput[i][0] *= output_[i][0] * (1.0 - output_[i][0]);
        } 
        return gradInput;
    }

};


class BinaryCrossEntropy {

public:

    double loss(double pred, double gt) {
        // We do not want to have 1.0 or 0.0 otherwise division by infinity
        pred = std::clamp(pred, EPS, 1.0 - EPS);
        return -(gt * std::log(pred) + (1.0 - gt) * std::log(1 - pred));
    }

    double dout(double pred, double gt) {
        // We do not want to have 1.0 or 0.0 otherwise division by infinity
        pred = std::clamp(pred, EPS, 1.0 - EPS);
        return -(gt / pred - (1.0 - gt) / (1.0 - pred));
    }
};


class Layer {

    int inputSize_;
    int outputSize_;
    
    std::vector<std::vector<double>> W; /* (output_dim, input_dim) */
    std::vector<std::vector<double>> b; /* (output_dim, 1) */
    std::vector<std::vector<double>> x; /* (input_dim, 1) stored for backtrack */

    std::vector<std::vector<double>> dW; /* (output_dim, input_dim) */
    std::vector<std::vector<double>> db; /* (output_dim, 1) */
    
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

    Layer(int inputSize, int outputSize) 
        : inputSize_{inputSize} , outputSize_{outputSize} {
            initializeWandB();
        }


    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& input
    ) {
        /*
        z = Wx + b
        a = activation_function(z)
        We have to store the input value for backtrack
        */
        x = input;
        std::vector<std::vector<double>> z = matMul(W, input); /*(output_dim, 1)*/
        for (int i = 0; i < outputSize_; ++i) {
            z[i][0] += b[i][0];
        }
        return z;
    }

    std::vector<std::vector<double>> backward(
        const std::vector<std::vector<double>>& dz
    ) {
        /*
        Compute derivatives and store them to update after each step
        Also compute how much input contributes to pass it to the prev layer
        So here we compute dW , db and dx , and we return dx to the prev layer
        */

        dW = matMul(dz, transposeMatrix(x));        // (output, 1) * (1, input)
        db = dz;                                    // (output, 1)

        std::vector<std::vector<double>> dx =
            matMul(transposeMatrix(W), dz);         // (input, output) * (output, 1)

        return dx;
        
    }

    void update() {
        /*Update W*/
        for (int i = 0; i < outputSize_; ++i) {
            for (int j = 0; j < inputSize_; ++j) {
                W[i][j] -= LR * dW[i][j];
            }
        }

        /*Update b*/
        for (int i = 0; i < outputSize_; ++i) {
            b[i][0] -= LR * db[i][0];
        }
    }

};


class Model {

    std::vector<Layer> layers_;
    std::vector<ReLU> relus_;
    Sigmoid sigmoid_;
    BinaryCrossEntropy loss_;

    int inputSize_ = 5;
    int outputSize_ = OUTPUTSIZE;

    void storeLayers(const std::vector<int>& layerSizes) {
        layers_.reserve(layerSizes.size() + 1);
        relus_.reserve(layerSizes.size()); // last one is sigmoid
        int prev = inputSize_;
        for (int size : layerSizes) {
            layers_.push_back(Layer(prev, size));
            relus_.push_back(ReLU());
            prev = size;
        }
        layers_.push_back(Layer(prev, outputSize_));
    }


    double forward_(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> x = input;
        // .size() returns an unsigned type std::size_t
        for (std::size_t i = 0; i < relus_.size(); ++i) {
            x = layers_[i].forward(x);
            x = relus_[i].forward(x);
        }
        x = layers_.back().forward(x);
        x = sigmoid_.forward(x);
        return x[0][0];
    }

public:

    Model(const std::vector<int>& layerSizes) {
        storeLayers(layerSizes);
    }

    double forward(const std::vector<std::vector<double>>& input) {
        return forward_(input);
    }
    


};


int test() {
    try {
        // Model with input size 5, hidden layers 8 and 4, output size 1
        Model model({8, 4});

        std::vector<std::vector<double>> x1 = {
            {1.0},
            {2.0},
            {3.0},
            {4.0},
            {5.0}
        };

        std::vector<std::vector<double>> x2 = {
            {-1.0},
            {0.5},
            {2.0},
            {-3.0},
            {1.5}
        };

        double pred1 = model.forward(x1);
        double pred2 = model.forward(x2);

        std::cout << "Prediction 1: " << pred1 << "\n";
        std::cout << "Prediction 2: " << pred2 << "\n";

        if (pred1 < 0.0 || pred1 > 1.0) {
            std::cout << "ERROR: pred1 is not a valid probability\n";
        }

        if (pred2 < 0.0 || pred2 > 1.0) {
            std::cout << "ERROR: pred2 is not a valid probability\n";
        }

        std::cout << "Class 1: " << (pred1 >= 0.5 ? 1 : 0) << "\n";
        std::cout << "Class 2: " << (pred2 >= 0.5 ? 1 : 0) << "\n";

    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }

    return 0;
}

int main() {
    return test();
}
