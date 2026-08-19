#include <iostream>
#include <random>
#include "autograd.cpp"
#include "Neuron.cpp"
#include "Layer.cpp"
#include "MLP.cpp"
#include 

struct DatoXOR {
    std::vector<float> x;
    float target;
};

int main() {
    std::cout << "=== ENTRENANDO LA IA (LOSS + SGD) ===\n\n";

    // 1. Instanciamos el generador aleatorio y la arquitectura
    std::mt19937 gen(1); 
    MLP modelo({2, 4, 4, 1}, gen); // Pasamos el generador para la optimización que aprendimos
    
    std::vector<DatoXOR> dataset = {
    {{0.0f, 0.0f}, 0.0f}, 
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f} 
};

    // 2. Extraemos parámetros
    std::vector<ValuePtr> parametros = modelo.parameters();
    std::cout << "Total de parametros a optimizar: " << parametros.size() << "\n\n";

    // 3. Dataset (Entrada y Objetivo)

    float learning_rate = 0.05f; 

    // BUCLE DE ENTRENAMIENTO
    for (int epoch = 1; epoch <= 1000; ++epoch) {
        for(const auto& aux : dataset){ 
        
	std::vector<ValuePtr> x = {make_value(aux.x[0]),
					 make_value(aux.x[1])};
	ValuePtr target = make_value(aux.target);
	
	// A) FORWARD PASS
        std::vector<ValuePtr> pred = modelo(x);
        ValuePtr resultado = pred[0];

        // B) PÉRDIDA (MSE)
        ValuePtr diff = resultado - target;
        ValuePtr loss = diff * diff;

        // C) ZERO GRAD
        for (auto& p : parametros) {
            p->grad = 0.0f; 
        }

        // D) BACKWARD PASS
        loss->grad = 1.0f;
        backward(loss);

        // E) OPTIMIZADOR SGDOB
        for (auto& p : parametros) {
            p->data = p->data - (learning_rate * p->grad);
        }

        // F) Progreso
        std::cout << "Epoch " << epoch << " | Loss: " << loss->data << " | Prediccion: " << resultado->data << "\n";
    }//endfor2
}//endfor1

    std::cout << "\n¡Entrenamiento completado! La red ha aprendido.\n";
    return 0;
}
