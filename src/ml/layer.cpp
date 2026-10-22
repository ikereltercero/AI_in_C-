#include "ml/Layer.h"
#include <cmath>
#include <omp.h>

Layer::Layer(size_t nin, size_t nout, std::mt19937& gen, bool is_linear) 
    : W(nin, nout), b(1, nout), is_linear(is_linear) {
    
    // Inicialización de pesos (He Initialization para ReLU)
    float scale = std::sqrt(2.0f / nin);
    std::normal_distribution<float> d(0.0f, scale);
    
    for (size_t i = 0; i < W.size(); ++i) {
        W.data[i] = d(gen);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b.data[i] = 0.0f; // Sesgos a 0
    }
}

Tensor Layer::forward(const Tensor& X) {
    X_cache = X; // Guardamos la entrada para el backward
    Tensor out(X.rows, W.cols);

    // Multiplicación de Matrices: Out = X * W + b
    #pragma omp parallel for
    for (int i = 0; i < (int)X.rows; ++i) {
        for (int j = 0; j < (int)W.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < (int)W.rows; ++k) {
                sum += X.at(i, k) * W.at(k, j);
            }
            
            float val = sum + b.at(0, j);
            
            // Función de Activación ReLU
            if (!is_linear && val < 0.0f) {
                val = 0.0f;
            }
            out.at(i, j) = val;
        }
    }
    
    out_cache = out; // Guardamos la salida para la derivada de ReLU
    return out;
}

Tensor Layer::backward(const Tensor& dOut) {
    Tensor dZ(dOut.rows, dOut.cols);

    // 1. Derivada de la activación (ReLU)
    #pragma omp parallel for
    for (int i = 0; i < (int)dOut.rows; ++i) {
        for (int j = 0; j < (int)dOut.cols; ++j) {
            if (!is_linear && out_cache.at(i, j) <= 0.0f) {
                dZ.at(i, j) = 0.0f;
            } else {
                dZ.at(i, j) = dOut.at(i, j);
            }
        }
    }

    // 2. Gradientes de los Pesos: dW = X^T * dZ
    #pragma omp parallel for
    for (int i = 0; i < (int)W.rows; ++i) {
        for (int j = 0; j < (int)W.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < (int)X_cache.rows; ++k) {
                sum += X_cache.at(k, i) * dZ.at(k, j);
            }
            // Acumulamos el gradiente (Adam lo usará luego)
            #pragma omp atomic
            W.grad[i * W.cols + j] += sum;
        }
    }

    // 3. Gradientes de los Sesgos: db = suma(dZ)
    for (int j = 0; j < (int)b.cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < (int)dZ.rows; ++i) {
            sum += dZ.at(i, j);
        }
        b.grad[j] += sum;
    }

    // 4. Gradiente para pasar a la capa anterior: dX = dZ * W^T
    Tensor dX(X_cache.rows, X_cache.cols);
    #pragma omp parallel for
    for (int i = 0; i < (int)dX.rows; ++i) {
        for (int j = 0; j < (int)dX.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < (int)W.cols; ++k) {
                sum += dZ.at(i, k) * W.at(j, k);
            }
            dX.at(i, j) = sum;
        }
    }

    return dX;
}
Tensor Layer::forward_inference(const Tensor& X) const {
    // NO HAY X_cache = X; (No guardamos estado, es solo lectura)
    Tensor out(X.rows, W.cols);

    //#pragma omp parallel for
    for (int i = 0; i < (int)X.rows; ++i) {
        for (int j = 0; j < (int)W.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < (int)W.rows; ++k) {
                sum += X.at(i, k) * W.at(k, j);
            }
            float val = sum + b.at(0, j);
            if (!is_linear && val < 0.0f) val = 0.0f; // ReLU
            out.at(i, j) = val;
        }
    }
    // NO HAY out_cache = out;
    return out;
}

  