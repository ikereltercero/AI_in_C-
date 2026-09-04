#include "Adam.h"
#include <cmath>
#include <omp.h>
Adam::Adam(const std::vector<ValuePtr>& params, float learning_rate) 
    : parametros(params), lr(learning_rate), beta1(0.9f), beta2(0.999f), epsilon(1e-8f), t(0) {
        for (auto& p : parametros) {
            estado.emplace(p.get(), Momentos{});
            
    }
    }

void Adam::zero_grad() {
    for (auto& p : parametros) {
        p->grad = 0.0f;
    }
}

void Adam::step() {
    t++; // Incrementamos el contador de época/paso

    const float beta1_t = std::pow(beta1, t);
    const float beta2_t = std::pow(beta2, t);

    const int n = static_cast<int>(parametros.size());
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        ValuePtr& p = parametros[i];
        Momentos& mom = estado.at(p.get()); 
        float grad = p->grad;

        // 1. Actualizamos medias móviles
        mom.m = beta1 * mom.m + (1.0f - beta1) * grad;
        mom.v = beta2 * mom.v + (1.0f - beta2) * (grad * grad);

        // 2. Corregimos el sesgo inicial
        float m_hat = mom.m / (1.0f - beta1_t);
        float v_hat = mom.v / (1.0f - beta2_t);

        // 3. Aplicamos el paso de optimización al peso real
        p->data = p->data - lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}