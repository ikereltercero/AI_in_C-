#include <iostream>
#include <random>
#include "autograd.h"
#include "Neuron.h"
#include "Layer.h"
#include "MLP.h"
#include <omp.h>
#include "Adam.h"
#include "DataLoader.h"

struct DatoXOR {
    std::vector<float> x;
    float target;
};

int argmax(const std::vector<ValuePtr>& predicciones) {
    int mejor_indice = 0;
    float max_val = predicciones[0]->data;
    
    for (size_t i = 1; i < predicciones.size(); ++i) {
        if (predicciones[i]->data > max_val) {
            max_val = predicciones[i]->data;
            mejor_indice = static_cast<int>(i);
        }
    }
    return mejor_indice;
}

int argmax_vector(const std::vector<float>& vec) {
    int mejor_indice = 0;
    float max_val = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            mejor_indice = static_cast<int>(i);
        }
    }
    return mejor_indice;
}

int main() {
    std::cout << "=== ENTRENANDO LA IA ===\n\n";

    DataLoader loader;
    if (!loader.cargarCSV("../dataset.csv", 4)) {
        std::cerr << "Abortando ejecución. Falló el DataLoader.\n";
        return -1;
    }
    const auto& dataset = loader.obtenerDatos();

    std::mt19937 gen(1); 
    MLP modelo({4, 8, 8, 3}, gen); 


    std::vector<ValuePtr> parametros = modelo.parameters();
    std::cout << "Total de parametros a optimizar: " << parametros.size() << "\n\n";


    float learning_rate = 0.001f; 
    Adam optimizador(parametros, learning_rate);
    std::cout << "=== INICIANDO ENTRENAMIENTO PARALELO CON OPENMP ===\n";
    
    // Opcional: Puedes forzar el número de hilos por código si lo deseas
    // omp_set_num_threads(4); 

    double t_inicio = omp_get_wtime();

    for (int epoch = 1; epoch <= 1000; ++epoch) {
        float loss_acumulada = 0.0f;
        int aciertos = 0;
        int total_muestras = 0;

        for (const auto& muestra : dataset) { 

            std::vector<ValuePtr> x; 
            std::vector<ValuePtr> target;
            for (int i =0; i < muestra.x.size(); i++){  
                x.push_back(make_value(muestra.x.at(i)));
            }
            for (int i =0; i < muestra.y.size(); i++){  
                target.push_back(make_value(muestra.y.at(i)));
            }        
            // A) FORWARD PASS
            std::vector<ValuePtr> pred = modelo(x);
            int index_pred = argmax(pred);
            int index = argmax_vector(muestra.y);
            if (index_pred == index) {
                aciertos++;
            }
            total_muestras++;

            // B) PÉRDIDA (MSE)
            ValuePtr loss(make_value(0.0f));
            for(int i = 0; i < pred.size(); i++){
                 ValuePtr diff = pred[i] - target[i];
                loss = loss + (diff * diff);
            }
           

            // C) ZERO GRAD
            optimizador.zero_grad();    

            // D) BACKWARD PASS
            backward(loss);

            // E) OPTIMIZADOR ADAM
            optimizador.step();

            loss_acumulada += loss->data;
        }

        if (epoch % 100 == 0) {
            float precision = (static_cast<float>(aciertos) / total_muestras) * 100.0f;
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << loss_acumulada 
                      << " | Aciertos: " << aciertos << "/" << total_muestras 
                      << " (" << precision << "%)\n";
        }
    }

    double t_fin = omp_get_wtime();
    double tiempo_total = t_fin - t_inicio;
    
    std::cout << "\n¡Entrenamiento finalizado!\n";
    printf("Tiempo total de ejecución: %.6f segundos.\n", tiempo_total);

    return 0;
}