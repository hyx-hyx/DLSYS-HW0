#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

/* X shape=(m,n) theta shape=(n,k)*/
void mat_mul(py::array_t<double> X, py::array_t<double> theta){
    auto buf_x = X.unchecked<2>();
    auto buf_theta = theta.unchecked<2>();
    
    int x_rows = buf_x.shape(0);
    int x_cols = buf_x.shape(1);
    int theta_rows = buf_theta.shape(0);
    int theta_cols = buf_theta.shape(1);
    if(x_cols!=theta_rows){
        throw std::runtime_error("Input a column not equal Input b row!");
    }

    auto result = py::array_t<float>({rows, cols});
    auto buf_result = result.mutable_unchecked<2>();

    for(size_t i=0;i<x_rows;++i){
        for(size_t k=0;k<theta_cols;++k){
            float sum=0;
            for(size_t j=0;j<x_cols;++j){
                sum+=X.at[i*x_cols+j]*theta[j*theta_cols+k];
            }
            result[i*x_cols+k]=sum;
        }
    }
    return result
}
void mat_transpose(float* x){
    for(size_t i=0;i<m;++i){
        for(size_t j=0;j<i;++j){
            float t=x[i*n+j];
            x[i*n+j]=x[j*n+i];
            x[j*n+i]=t;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
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
    auto x_batch = py::array_t<float>({batch, n});
    float y_batch = py::array_t<float>({m});
    for(size_t i=0;i<m;i+=batch){
        size_t end=i+batch<m?i+batch:m;
        size_t b=end-i;
        for(size_t idx=i;idx<i+b;++idx)
            X_batch[idx-i]=X[idx];
        for(size_t idx=i;idx<i+b;++idx)
            y_batch[idx-i]=y[idx];
        
        mat_mul(theta_t_x,X_batch,y_batch,m,n,k);

        // exp(theta_t_x)
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
