
#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <charconv>    // Para std::from_chars (Parsing ultra-rápido)
#include <string_view> // Para leer texto sin hacer copias de memoria

bool DataLoader::cargarCSV(const std::string& ruta, size_t num_entradas) {
    std::ifstream archivo(ruta);
    if (!archivo) {
        std::cerr << "Error al abrir: " << ruta << '\n';
        return false;
    }

    dataset.clear();
    dataset.reserve(10000); 

    std::string linea;
    size_t numero_linea = 0;
    while (std::getline(archivo, linea)) {
        ++numero_linea;
        if (linea.empty()) continue;

        Muestra muestra;
        muestra.x.reserve(num_entradas); 
        
        std::string_view sv(linea); 
        size_t col = 0;
        bool fila_valida = true;


        while (!sv.empty()) {
            size_t pos = sv.find(',');
            std::string_view token = (pos == std::string_view::npos) ? sv : sv.substr(0, pos);

            float val;
            auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), val);

            if (ec != std::errc() || std::isnan(val) || std::isinf(val)) {
                fila_valida = false;
                break; 
            }

            if (col < num_entradas) muestra.x.push_back(val);
            else muestra.y.push_back(val);
            
            col++;

            if (pos == std::string_view::npos) break; 
            sv = sv.substr(pos + 1); 
        }

        if (!fila_valida || muestra.x.size() != num_entradas || muestra.y.empty()) {
            continue; 
        }

        dataset.push_back(std::move(muestra));
    }

    std::cout << "Dataset cargado rápido. Muestras: " << dataset.size() << '\n';
    return true;
}
const std::vector<Muestra>& DataLoader::obtenerDatos() const {
    return dataset;
}

size_t DataLoader::tamano() const {
    return dataset.size();
}