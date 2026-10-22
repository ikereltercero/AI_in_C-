#pragma once
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <algorithm>


class Tensor {
public:
    size_t rows;
    size_t cols;
    std::vector<float> data;
    std::vector<float> grad; 
    

    Tensor() : rows(0), cols(0) {}
    Tensor(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f), grad(r * c, 0.0f) {}

    float& at(size_t r, size_t c) {
        return data[r * cols + c];
    }
    const float& at(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    size_t size() const { return rows * cols; }
    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
};