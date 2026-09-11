#include "../include/Adam.h"
#include <cmath>
#include <stdexcept>



Adam::Adam(const std::vector<Tensor*>& params, float learning_rate) 
    : parametros(params), lr(learning_rate), beta1(0.9f), beta2(0.999f), epsilon(1e-8f), t(0) {
    if (learning_rate <= 0.0f) {
    throw std::invalid_argument("Learning rate must be positive");
    }

    // Inicializamos la memoria de momentos (m y v) con ceros exactos al tamaño de cada tensor
    m.reserve(parametros.size());
    v.reserve(parametros.size());
    for (Tensor* p : parametros) {
        if (p == nullptr) throw std::invalid_argument("Adam received a null Tensor");

        m.push_back(std::vector<float>(p->size(), 0.0f));
        v.push_back(std::vector<float>(p->size(), 0.0f));
    }
}

void Adam::zero_grad() {
    for (Tensor* p : parametros) {
        p->zero_grad();
    }
}

void Adam::step() {
    ++t;

    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);

    // Estos cálculos se hacen una vez por step, no por cada peso
    const float lr_t = lr * std::sqrt(1.0f - beta2_t)
                    / (1.0f - beta1_t);

    for (size_t i = 0; i < parametros.size(); ++i) {
        Tensor* p = parametros[i];
        
        // Acceso secuencial a memoria para favorecer la localidad de caché.
        for (size_t j = 0; j < p->size(); ++j) {
            const float grad = p->grad[j];
            
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * grad;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * grad * grad;
            
            p->data[j] -= lr_t * m[i][j] / (std::sqrt(v[i][j]) + epsilon);
        }
    }
}