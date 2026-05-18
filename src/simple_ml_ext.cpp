#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <iostream>

namespace py = pybind11;

/* X shape=(m,n) theta shape=(n,k)*/
void mat_mul(float* result, const float* X, float* theta, size_t m, size_t n,
             size_t k) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      float sum = 0;
      for (size_t j = 0; j < n; ++j) {
        sum += X[i * n + j] * theta[j * k + k_idx];
      }
      result[i * k + k_idx] = sum;
    }
  }
}

void transponse(float* result, const float* X, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[j * m + i] = X[i * n + j];
    }
  }
}
void softmax_regression_epoch_cpp(const float* X, const unsigned char* y,
                                  float* theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  float X_batch[batch * n];
  float y_batch[batch];
  float x_theta[batch * k];

  for (size_t i = 0; i < m; i += batch) {
    size_t end = i + batch < m ? i + batch : m;
    size_t b = end - i;
    size_t x_step_size = n;
    for (size_t idx = i * x_step_size; idx < (i + b) * x_step_size; ++idx)
      X_batch[idx - (i * x_step_size)] = X[idx];
    for (size_t idx = i; idx < i + b; ++idx) y_batch[idx - i] = y[idx];
    mat_mul(x_theta, X_batch, theta, b, n, k);

    for (size_t i = 0; i < b * k; ++i) {
      std::cout << "x_theta " << i << " " << x_theta[i] << std::endl;
    }

    // exp(theta_t_x) and normalize
    float exp_sum = 0;
    for (size_t i = 0; i < b * k; ++i) {
      x_theta[i] = exp(x_theta[i]);
      if (i % k == 0 && i > 0) {
        for (size_t j = i - k; j < i; ++j) {
          x_theta[j] /= exp_sum;
        }
        exp_sum = 0;
      }
      exp_sum += x_theta[i];
    }
    for (size_t j = (b - 1) * k; j < b * k; ++j) {
      x_theta[j] /= exp_sum;
    }

    // Z-Iy
    for (size_t i = 0; i < b; ++i) {
      x_theta[i * k + int(y_batch[i])] -= 1;
    }
    float grad[n * k];
    float X_batch_transpose[n * b];
    transponse(X_batch_transpose, X_batch, b, n);
    mat_mul(grad, X_batch_transpose, x_theta, n, b, k);

    for (size_t i = 0; i < n * k; ++i) {
      std::cout << grad[i] << " ";
      grad[i] /= b;
      theta[i] -= lr * grad[i];
    }
  }

  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
