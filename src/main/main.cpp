#include <iostream>
#include <random>
#include <omp.h>
#include <cstdio>
#include "ml/DataLoader.h"
#include "ml/MLP.h"
#include "ml/Adam.h"
#include "ml/Loss.h"

template<LossFunction LossType>
void entrenar_modelo_con_evaluacion(MLP& modelo, DataLoader& loader, Adam& optimizador, int num_epochs, size_t batch_size) {
    
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        loader.shuffleTrain();
        loader.resetTrain();
        float loss_epoca = 0.0f;

        while (true) {
            BatchView batch = loader.next_train_batch(batch_size);
            if (batch.samples == 0) break;

            optimizador.zero_grad();

            Tensor pred = modelo.forward(batch);
            float batch_loss = LossType::forward(pred, batch.Y_ptr, batch.num_outputs);
            Tensor dLoss = LossType::backward(pred, batch.Y_ptr, batch.num_outputs);
            
            loss_epoca += batch_loss;

            modelo.backward(dLoss);
            optimizador.step();
        }

        // Evaluación periódica en Test (para ver si realmente acierta las flores)
        if (epoch % 100 == 0) {
            loader.resetTest();
            size_t aciertos = 0;
            size_t total = 0;

            while (true) {
                BatchView test_batch = loader.next_test_batch(batch_size);
                if (test_batch.samples == 0) break;

                Tensor pred_test = modelo.forward(test_batch);

                for (size_t i = 0; i < pred_test.rows; ++i) {
                    // Argmax de la predicción de la IA
                    int idx_pred = 0;
                    float max_val = pred_test.at(i, 0);
                    for (size_t j = 1; j < pred_test.cols; ++j) {
                        if (pred_test.at(i, j) > max_val) {
                            max_val = pred_test.at(i, j);
                            idx_pred = static_cast<int>(j);
                        }
                    }

                    // Argmax del valor real (One-Hot)
                    int idx_real = 0;
                    float max_real = test_batch.Y_ptr[i * test_batch.num_outputs + 0];
                    for (size_t j = 1; j < test_batch.num_outputs; ++j) {
                        float val = test_batch.Y_ptr[i * test_batch.num_outputs + j];
                        if (val > max_real) {
                            max_real = val;
                            idx_real = static_cast<int>(j);
                        }
                    }

                    if (idx_pred == idx_real) aciertos++;
                    total++;
                }
            }

            float acc = (static_cast<float>(aciertos) / total) * 100.0f;
            std::printf("Epoch %4d | CrossEntropy Loss: %.4f | Test Accuracy: %zu/%zu (%.2f%%)\n", 
                        epoch, loss_epoca, aciertos, total, acc);
        }
    }
}


int main() {
    std::cout << "=== ENTRENANDO MOTOR V2 (TENSORS & MATRICES) ===\n\n";

    DataLoader loader(42);
    // Asumimos 4 entradas y 3 salidas según tu dataset de Iris (One-Hot Encoding)
    if (!loader.cargarCSV("../dataset.csv", 4, 3)) {
        std::cerr << "Abortando ejecución. Falló el DataLoader.\n";
        return -1;
    }

    loader.split_train_test(0.8f);

    std::cout << "Muestras de Train: " << loader.tamanoTrain() << '\n';
    std::cout << "Muestras de Test: " << loader.tamanoTest() << "\n\n";

    std::mt19937 gen(1);
    MLP modelo({4, 8, 8, 3}, gen);

    float learning_rate = 0.0005f;
    Adam optimizador(modelo.parameters(), learning_rate);

    std::cout << "=== INICIANDO ENTRENAMIENTO MOTOR V2 ===\n";
    double t_inicio = omp_get_wtime();

    // Entrenamos inyectando el tipo de Loss que queramos:
    entrenar_modelo_con_evaluacion<CrossEntropy>(modelo, loader, optimizador, 1000, 16);
    
    // Si quisieras usar MSE, simplemente harías:
    // entrenar_modelo<MSE>(modelo, loader, optimizador, 1000, 16);

    double t_fin = omp_get_wtime();
    std::printf("\nTiempo total de ejecución: %.6f segundos.\n", t_fin - t_inicio);
    
    // Serialización segura
    modelo.save("../modelo_v2_iris.bin");

    return 0;
}