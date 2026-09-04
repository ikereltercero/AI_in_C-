#pragma once
#include <array>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include <cstdint>


struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    float data;
    float grad;
    std::array<ValuePtr, 2> prev{};
    uint8_t prev_count = 0;
    std::function<void(float)> backward_op;

    explicit Value(float val);
};

ValuePtr make_value(float val);

ValuePtr operator+(const ValuePtr& x, const ValuePtr& y);
ValuePtr operator*(const ValuePtr& x, const ValuePtr& y);
ValuePtr operator-(const ValuePtr& x, const ValuePtr& y);
ValuePtr operator/(const ValuePtr& x, const ValuePtr& y);

ValuePtr relu(const ValuePtr& x);
void backward(const ValuePtr& root);