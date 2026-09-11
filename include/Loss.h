#pragma once
#include "Tensor.h"
#include <cmath>
#include <algorithm>

// 1. Definimos el Concept: Toda función de Loss debe tener un forward y un backward con esta firma exacta.
template<typename T>
concept LossFunction = requires(T, const Tensor& pred, const float* target, size_t num_outputs) {
    { T::forward(pred, target, num_outputs) } -> std::convertible_to<float>;
    { T::backward(pred, target, num_outputs) } -> std::same_as<Tensor>;
};

// ==========================================
// MEAN SQUARED ERROR (Para regresión o XOR)
// ==========================================
struct MSE {
    static float forward(const Tensor& pred, const float* Y_batch, size_t num_outputs) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < pred.rows; ++i) {
            for (size_t j = 0; j < pred.cols; ++j) {
                float diff = pred.at(i, j) - Y_batch[i * num_outputs + j];
                total_loss += diff * diff;
            }
        }
        return total_loss / pred.rows;
    }

    static Tensor backward(const Tensor& pred, const float* Y_batch, size_t num_outputs) {
        Tensor dLoss(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.rows; ++i) {
            for (size_t j = 0; j < pred.cols; ++j) {
                float diff = pred.at(i, j) - Y_batch[i * num_outputs + j];
                dLoss.at(i, j) = (2.0f * diff) / pred.rows;
            }
        }
        return dLoss;
    }
};

// ==========================================
// CROSS-ENTROPY + SOFTMAX (Para clasificación multiclase como Iris)
// ==========================================
struct CrossEntropy {
    static float forward(const Tensor& pred, const float* Y_batch, size_t num_outputs) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < pred.rows; ++i) {
            // Truco de estabilidad numérica: restar el máximo antes del exp()
            float max_val = pred.at(i, 0);
            for (size_t j = 1; j < pred.cols; ++j) {
                max_val = std::max(max_val, pred.at(i, j));
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < pred.cols; ++j) {
                sum_exp += std::exp(pred.at(i, j) - max_val);
            }

            for (size_t j = 0; j < pred.cols; ++j) {
                if (Y_batch[i * num_outputs + j] == 1.0f) { // Solo nos importa la clase real (One-Hot)
                    float softmax_prob = std::exp(pred.at(i, j) - max_val) / sum_exp;
                    // Evitar log(0) sumando un epsilon
                    total_loss += -std::log(softmax_prob + 1e-8f);
                }
            }
        }
        return total_loss / pred.rows;
    }

    static Tensor backward(const Tensor& pred, const float* Y_batch, size_t num_outputs) {
        Tensor dLoss(pred.rows, pred.cols);
        
        for (size_t i = 0; i < pred.rows; ++i) {
            float max_val = pred.at(i, 0);
            for (size_t j = 1; j < pred.cols; ++j) {
                max_val = std::max(max_val, pred.at(i, j));
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < pred.cols; ++j) {
                sum_exp += std::exp(pred.at(i, j) - max_val);
            }

            // Derivada mágica y simplificada de Softmax + CrossEntropy: (Probabilidad - ValorReal) / BatchSize
            for (size_t j = 0; j < pred.cols; ++j) {
                float softmax_prob = std::exp(pred.at(i, j) - max_val) / sum_exp;
                float y_true = Y_batch[i * num_outputs + j];
                dLoss.at(i, j) = (softmax_prob - y_true) / pred.rows;
            }
        }
        return dLoss;
    }
};