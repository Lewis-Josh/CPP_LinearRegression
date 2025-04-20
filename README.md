# CPP_LinearRegression

# Linear Regression Model

This project implements a simple Linear Regression model in C++ with support for L2 regularization, learning rate decay, and model persistence (saving and loading). The model is designed to train on a dataset and make predictions for regression tasks.

---

## Features

- **Gradient Descent Optimization**: Uses gradient descent to minimize the loss function.
- **L2 Regularization**: Prevents overfitting by penalizing large weights.
- **Learning Rate Decay**: Gradually reduces the learning rate during training for better convergence.
- **Model Persistence**: Save and load model parameters to/from a file.
- **Mean Squared Error (MSE)**: Evaluate the model's performance.

---

## Requirements

- **C++ Compiler**: Supports C++14 or later.
- **Standard Libraries**: `<vector>`, `<memory>`, `<iostream>`, `<fstream>`, `<stdexcept>`, `<cmath>`.

---

## How to Compile and Run

### Compilation
Use a C++ compiler to compile the project. For example, with `g++`:

---

## Usage

### Example Input
The `main` function demonstrates how to use the `LinearRegression` class:
1. Create an instance of the model:

After training, predictions and the final MSE are displayed:

---

## File Structure

- **`LinearRegression.h`**: Header file containing the class definition.
- **`LinearRegression.cpp`**: Implementation of the Linear Regression model.
- **`linear_regression_model.txt`**: File to save/load model parameters.

---

## Known Issues

- The model assumes the input data (`X` and `y`) is preprocessed and normalized.
- The current implementation does not handle categorical features or missing data.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
