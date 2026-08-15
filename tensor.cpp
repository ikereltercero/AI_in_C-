#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono> // Para medir tiempos exactos (El time.time() de C++)
#include <random> // Para inicializar pesos aleatorios

class Tensor {
private:
    size_t rows;
    size_t cols;
    std::vector<float> data;

public:
    // Constructor
    Tensor(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    // Rellenar con números aleatorios (Simulando pesos de una Red Neuronal)
    void randomize() {
        std::mt19937 gen(42); // Semilla fija para reproducibilidad
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (float& val : data) {
            val = dis(gen);
        }
    }

    // Getter (Implícitamente inline, como bien dijiste)
    float& get(size_t i, size_t j) { return data[i * cols + j]; }
    const float& get(size_t i, size_t j) const { return data[i * cols + j]; }

    // SOBRECARGA DE OPERADORES: Magia para poder hacer C = A + B
    Tensor operator+(const Tensor& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Error de Shape en Broadcasting");
        }
        Tensor result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    // MULTIPLICACIÓN MATRICIAL INGENUA (El estándar matemático I-J-K)
    Tensor matmul_naive(const Tensor& W) const {
        if (cols != W.rows) throw std::invalid_argument("Shapes incompatibles para Matmul");
        
        Tensor Z(rows, W.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < W.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    // W.get(k, j) salta columnas enteras en memoria. ¡Destruye la caché L1!
                    sum += this->get(i, k) * W.get(k, j);
                }
                Z.get(i, j) = sum;
            }
        }
        return Z;
    }

    // MULTIPLICACIÓN MATRICIAL OPTIMIZADA (Orden I-K-J)
    Tensor matmul_optimized(const Tensor& W) const {
        if (cols != W.rows) throw std::invalid_argument("Shapes incompatibles");
        
        Tensor Z(rows, W.cols);
        // Cambiamos el orden de los bucles. Ahora la memoria se lee secuencialmente.
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                float x_ik = this->get(i, k); // Extraemos la constante del bucle interno
                for (size_t j = 0; j < W.cols; ++j) {
                    // Memoria 100% contigua. El procesador usa instrucciones SIMD automáticamente.
                    Z.get(i, j) += x_ik * W.get(k, j);
                }
            }
        }
        return Z;
    }
};// end class

int main() {
    std::cout << "=== PROFILING DE CACHÉ L1 EN C++ ===\n";
    
    // Matrices de 1000x1000 (1 millón de floats = 4MB cada una)
    size_t N = 1000;
    Tensor X(N, N);
    Tensor W(N, N);
    X.randomize();
    W.randomize();

    std::cout << "Multiplicando matrices (" << N << "x" << N << ")...\n\n";

    // Medir código Naive
    auto start = std::chrono::high_resolution_clock::now();
    Tensor Z_naive = X.matmul_naive(W);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_naive = end - start;
    std::cout << "1. Matmul Naive (Orden I-J-K): " << duration_naive.count() << " ms\n";

    // Medir código Optimizado para Caché
    start = std::chrono::high_resolution_clock::now();
    Tensor Z_opt = X.matmul_optimized(W);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_opt = end - start;
    std::cout << "2. Matmul Cache-Opt (Orden I-K-J): " << duration_opt.count() << " ms\n";

    std::cout << "\nMejora de velocidad: " << duration_naive.count() / duration_opt.count() << "x más rápido SOLO por cambiar 2 líneas de sitio.\n";

    return 0;
}