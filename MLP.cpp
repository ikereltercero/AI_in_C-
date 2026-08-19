#include <vector>
#include "Layer.cpp"
#include <fstream>
#include <stdexcept>
#include <string>
class MLP {
private:
    std::vector<Layer> layers;
    
public:
    // Constructor: Recibe una lista con los tamaños de todas las capas
   MLP(const std::vector<size_t>& sizes, std::mt19937& gen) {
        layers.reserve(sizes.size() - 1);
        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            layers.emplace_back(sizes[i], sizes[i+1], gen, i == sizes.size() - 2); 
        }
    }
    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params;
        for (auto& layer : layers) {
            auto layer_params = layer.parameters();
            // Concatenamos los parámetros de la capa a la lista global
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    void save(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("No se pudo crear el archivo.");
        
        auto params = parameters();
        for (auto& p : params) {
            out.write(reinterpret_cast<const char*>(&p->data), sizeof(float));
        }
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("No se encontró el archivo.");

        auto params = parameters();
        for (auto& p : params) {
            in.read(reinterpret_cast<char*>(&p->data), sizeof(float));
        }
    }

    // ==========================================
    // TU ÚLTIMA MISIÓN ARQUITECTÓNICA
    // ==========================================
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x) {
        
        // TODO: Implementar el Forward Pass secuencial
        // 1. Tienes los datos originales 'x'.
        // 2. Debes pasárselos a la primera capa, coger el resultado, 
        //    pasárselo a la segunda capa, coger el resultado...
        // 3. Devuelve el vector resultante de la última capa.

        // ¡Tu código aquí!
	std::vector<ValuePtr> aux = x;
	bool is_linear;
	for(size_t i=0; i<layers.size(); i++){
	  aux = layers[i](aux);
}
	return aux;
    }

};
