#include <iostream>
#include <random>
#include "autograd.cpp"
 #include "Neuron.cpp"
 #include "Layer.cpp"
 #include "MLP.cpp"

int main() {
    std::cout << "=== ENTRENANDO LA IA (LOSS + SGD) ===\n\n";

    // 1. Instanciamos el generador aleatorio y la arquitectura
    std::mt19937 gen(1); 
    MLP modelo({3, 4, 4, 1}, gen); // Pasamos el generador para la optimización que aprendimos

    // 2. Extraemos parámetros
    std::vector<ValuePtr> parametros = modelo.parameters();
    std::cout << "Total de parametros a optimizar: " << parametros.size() << "\n\n";

    // 3. Dataset (Entrada y Objetivo)
    std::vector<ValuePtr> x = {make_value(2.0f), make_value(3.0f), make_value(-1.0f)};
    ValuePtr target = make_value(1.0f); 

    float learning_rate = 0.05f; 

    // BUCLE DE ENTRENAMIENTO
    for (int epoch = 1; epoch <= 20; ++epoch) {
        
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

        // E) OPTIMIZADOR SGD
        for (auto& p : parametros) {
            p->data = p->data - (learning_rate * p->grad);
        }

        // F) Progreso
        std::cout << "Epoch " << epoch << " | Loss: " << loss->data << " | Prediccion: " << resultado->data << "\n";
    }

    std::cout << "\n¡Entrenamiento completado! La red ha aprendido.\n";
    return 0;
}
