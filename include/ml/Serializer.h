#pragma once
#include <fstream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "ml/MLP.h"
#include <cstdint>


const std::string MLP_MAGIC = "MODL"; 
const int MLP_VERSION = 2;
// 1. Concept para tipos planos
template<typename T>
concept POD = std::is_trivially_copyable_v<T>;

// 2. Escritura/Lectura de tipos planos
template<POD T>
void writeBinary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("Error escribiendo dato POD.");
}

template<POD T>
void readBinary(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("Error leyendo dato POD.");
}

// 3. Escritura/Lectura de vectores dinámicos
template<POD T>
void writeBinary(std::ofstream& out, const std::vector<T>& vec) {
    size_t sz = vec.size();
    writeBinary(out, sz); 
    out.write(reinterpret_cast<const char*>(vec.data()), sz * sizeof(T));
    if (!out) throw std::runtime_error("Error escribiendo vector.");
}

template<POD T>
void readBinary(std::ifstream& in, std::vector<T>& vec) {
    size_t sz;
    readBinary(in, sz); 
    vec.resize(sz);     
    in.read(reinterpret_cast<char*>(vec.data()), sz * sizeof(T));
    if (!in) throw std::runtime_error("Error leyendo vector.");
}