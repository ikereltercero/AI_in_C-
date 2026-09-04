#include "../include/autograd.h"

Value::Value(float val) : data(val), grad(0.0f), backward_op([](float) {}) {}

ValuePtr make_value(float val) {
    return std::make_shared<Value>(val);
}

ValuePtr operator+(const ValuePtr& x, const ValuePtr& y) {
    ValuePtr out = make_value(x->data + y->data);
    out->prev[0] = x;
    out->prev[1] = y;
    out->prev_count = 2;

    Value* x_ptr = x.get();
    Value* y_ptr = y.get();
    out->backward_op = [x_ptr, y_ptr](float grad) {
        x_ptr->grad += grad;
        y_ptr->grad += grad;
    };
    return out;
}

ValuePtr operator*(const ValuePtr& x, const ValuePtr& y) {
    ValuePtr out = make_value(x->data * y->data);
    out->prev[0] = x;
    out->prev[1] = y;
    out->prev_count = 2;

    Value* x_ptr = x.get();
    Value* y_ptr = y.get();
    out->backward_op = [x_ptr, y_ptr](float grad) {
        x_ptr->grad += y_ptr->data * grad;
        y_ptr->grad += x_ptr->data * grad;
    };
    return out;
}

ValuePtr operator-(const ValuePtr& x, const ValuePtr& y) {
    ValuePtr out = make_value(x->data - y->data);
    out->prev[0] = x;
    out->prev[1] = y;
    out->prev_count = 2;

    Value* x_ptr = x.get();
    Value* y_ptr = y.get();
    out->backward_op = [x_ptr, y_ptr](float grad) {
        x_ptr->grad += grad;
        y_ptr->grad -= grad;
    };
    return out;
}

ValuePtr operator/(const ValuePtr& x, const ValuePtr& y) {
    ValuePtr out = make_value(x->data / y->data);
    out->prev[0] = x;
    out->prev[1] = y;
    out->prev_count = 2;

    Value* x_ptr = x.get();
    Value* y_ptr = y.get();
    out->backward_op = [x_ptr, y_ptr](float grad) {
        x_ptr->grad += grad / y_ptr->data;
        y_ptr->grad -= (x_ptr->data / (y_ptr->data * y_ptr->data)) * grad;
    };
    return out;
}

ValuePtr relu(const ValuePtr& x) {
    ValuePtr out = make_value(x->data > 0.0f ? x->data : 0.0f);
    out->prev[0] = x;
    out->prev_count = 1;

    Value* x_ptr = x.get();
    const float local_grad = x->data > 0.0f ? 1.0f : 0.0f;
    out->backward_op = [x_ptr, local_grad](float grad) {
        x_ptr->grad += local_grad * grad;
    };
    return out;
}

void backward(const ValuePtr& root) {
    std::vector<ValuePtr> topo;
    std::unordered_set<Value*> visited;
    topo.reserve(64);
    visited.reserve(64);

    std::function<void(const ValuePtr&)> build_topo = [&](const ValuePtr& v) {
        if (!visited.insert(v.get()).second) return;
        for (uint8_t i = 0; i < v->prev_count; ++i) {
            build_topo(v->prev[i]);
        }
        topo.push_back(v);
    };

    build_topo(root);
    root->grad = 1.0f;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        ValuePtr& v = *it;
        v->backward_op(v->grad);
    }
}