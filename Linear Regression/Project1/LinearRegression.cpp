#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <stdexcept> // For exception handling
#include <cmath>     // For pow

LinearRegression::LinearRegression(double LR_bias, double LR_learning_rate, int LR_num_iterations, double LR_lambda)
    : weight(std::make_unique<std::vector<double>>()), // Initialize smart pointer
    bias(LR_bias),                                     // Initialize bias
    learning_rate(LR_learning_rate),                   // Initialize learning rate
    num_iterations(LR_num_iterations),                 // Initialize number of iterations
    lambda(LR_lambda)                                  // Initialize regularization strength
{
}

void LinearRegression::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) {
    // Validate input dimensions
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Input data dimensions are invalid.");
    }
    for (const auto& row : X) {
        if (row.size() != X[0].size()) {
            throw std::invalid_argument("Inconsistent feature dimensions in input data.");
        }
    }

    // Initialize weights if not already initialized
    if (weight->empty()) {
        weight->resize(X[0].size(), 0.0);
    }

    // Training loop
    for (int i = 0; i < this->num_iterations; i++) {
        // Compute the gradients
        std::vector<double> dw(X[0].size(), 0.0);
        double db = 0.0;

        for (size_t j = 0; j < X.size(); j++) {
            double y_hat = this->bias;
            for (size_t k = 0; k < X[j].size(); k++) {
                y_hat += (*this->weight)[k] * X[j][k];
            }

            double error = y[j][0] - y_hat;
            db += error;
            for (size_t k = 0; k < X[j].size(); k++) {
                dw[k] += error * X[j][k];
            }
        }

        // Update the weights and bias with L2 regularization
        for (size_t j = 0; j < weight->size(); j++) {
            (*this->weight)[j] += this->learning_rate * (dw[j] / X.size() - this->lambda * (*this->weight)[j]);
        }
        this->bias += this->learning_rate * db / X.size();

        // Apply learning rate decay
        this->learning_rate *= (1.0 / (1.0 + 0.01 * i)); // Decay rate = 0.01

        // Calculate and print loss (MSE) for the current epoch
        double mse = this->mean_squared_error(y, this->predict(X));
        std::cout << "Epoch " << i + 1 << "/" << this->num_iterations
            << " completed. Loss: " << mse << std::endl;
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const {
    // Validate input dimensions
    if (X.empty() || weight->empty()) {
        throw std::invalid_argument("Model is not trained or input data is invalid.");
    }

    std::vector<double> predictions(X.size(), 0.0);

    for (size_t i = 0; i < X.size(); i++) {
        double y_hat = bias;
        for (size_t j = 0; j < X[i].size(); j++) {
            y_hat += (*weight)[j] * X[i][j];
        }
        predictions[i] = y_hat;
    }

    return predictions;
}

double LinearRegression::mean_squared_error(const std::vector<std::vector<double>>& y_true, const std::vector<double>& y_pred) const {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Mismatched dimensions between true and predicted values.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        mse += std::pow(y_true[i][0] - y_pred[i], 2);
    }
    return mse / y_true.size();
}

void LinearRegression::save_model(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving model.");
    }

    file << bias << "\n";
    for (const auto& w : *weight) {
        file << w << "\n";
    }
    file.close();
}

void LinearRegression::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading model.");
    }

    file >> bias;
    weight->clear();
    double w;
    while (file >> w) {
        weight->push_back(w);
    }
    file.close();
}

int main() {
    // Create an instance of LinearRegression
    LinearRegression model(0.0, 0.1, 100, 0.01); // Added lambda for regularization

    // Example input data
    std::vector<std::vector<double>> X = { {1.0, 2.0}, {3.0, 4.0} };
    std::vector<std::vector<double>> y = { {5.0}, {6.0} };

    // Train the model
    model.train(X, y);

    // Predict using the trained model
    std::vector<double> predictions = model.predict(X);

    // Print predictions
    for (const auto& pred : predictions) {
        std::cout << pred << std::endl;
    }

    // Evaluate the model
    double mse = model.mean_squared_error(y, predictions);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // Save the model
    model.save_model("linear_regression_model.txt");

    // Load the model
    LinearRegression loaded_model;
    loaded_model.load_model("linear_regression_model.txt");

    return 0;
}