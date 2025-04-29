#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#pragma once

#include <vector>
#include <memory> // For smart pointers
#include <string> // For file operations

class LinearRegression {
public:
    // Constructor with configurable parameters
    LinearRegression(double bias = 0.05, double learning_rate = 0.10, int num_iterations = 10, double lambda = 0.0);

    // Train the model
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y);

    // Predict the output for the given input
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    // Evaluate the model using Mean Squared Error (MSE)
    double mean_squared_error(const std::vector<std::vector<double>>& y_true, const std::vector<double>& y_pred) const;

    // Save the model parameters to a file
    void save_model(const std::string& filename) const;

    // Load the model parameters from a file
    void load_model(const std::string& filename);

private:
    // Model parameters
    std::unique_ptr<std::vector<double>> weight; // Dynamically managed weights
    double bias;                                 // Bias term
    double learning_rate;                        // Learning rate
    int num_iterations;                          // Number of iterations
    double lambda;                               // Regularization strength (L2 regularization)
};

#endif // LINEARREGRESSION_H