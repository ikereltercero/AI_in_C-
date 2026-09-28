#pragma once
#include <vector>
#include <random>
#include "Layer.h"
#include "DataLoader.h" // Para recibir el BatchView

class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(const std::vector<size_t>& sizes, std::mt19937& gen);
    
    // Devuelve referencias directas a los tensores para el Optimizador
    std::vector<Tensor*> parameters();
    
    // Convierte el BatchView (memoria de DataLoader) en un Tensor inicial y lo propaga
    Tensor forward(const BatchView& batch);
    
    // Propaga el gradiente de la función de pérdida hacia atrás
    void backward(const Tensor& dLoss);

    void save(const std::string& filename);
    void load(const std::string& filename);

    Tensor inference(const BatchView& batch) const;
};