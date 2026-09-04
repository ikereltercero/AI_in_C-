#pragma once
#include <vector>
#include <string>

struct Muestra {
    std::vector<float> x;
    std::vector<float> y; // Permite soportar múltiples salidas si lo necesitamos

     Muestra() = default;
     Muestra(std::vector<float> x, std::vector<float> y)
        : x(std::move(x)), y(std::move(y)) {}
};

class DataLoader {
private:
    std::vector<Muestra> dataset;

public:
    // Constructor que lee directamente el archivo CSV
    bool cargarCSV(const std::string& ruta, size_t num_entradas);

    // Métodos para acceder a los datos
    const std::vector<Muestra>& obtenerDatos() const;
    size_t tamano() const;
};
