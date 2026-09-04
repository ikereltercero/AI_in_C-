# Autograd Engine & MLP in C++

Un motor de diferenciación automática (Autograd) y una implementación de un Perceptrón Multicapa (MLP) construidos completamente desde cero en C++ moderno. 

Inspirado en la arquitectura interna de PyTorch (motor autograd con grafos acíclicos dirigidos), este proyecto fue creado para comprender a bajo nivel las matemáticas y la gestión de memoria detrás del Deep Learning.

## Características Técnicas

* **Diferenciación Automática Inversa (Backpropagation):** Implementación de un Grafo Acíclico Dirigido (DAG) que rastrea operaciones matemáticas dinámicamente.
* **Ordenamiento Topológico:** Algoritmo de ordenamiento DFS (Depth-First Search) para calcular gradientes de forma eficiente y segura.
* **Gestión de Memoria Segura (RAII):** Uso avanzado de `std::shared_ptr`, mitigación de ciclos de referencia (Memory Leaks) mediante captura de punteros crudos (`.get()`) en funciones lambda, y optimización de memoria contigua con `std::array`.
* **Red Neuronal (MLP):** Clases estructuradas para `Neuron`, `Layer` y `MLP`.
* **Optimizador SGD:** Descenso de Gradiente Estocástico para el ajuste de pesos.

##  Tecnologías

* **C++17**
* Standard Template Library (STL)
* No se han utilizado librerías externas de matemáticas o IA.

##  Cómo compilar y ejecutar

Puedes compilar el proyecto utilizando `g++` (requiere soporte para C++17):

```bash
g++ -std=c++17 -O3 main.cpp -o ia_motor
./ia_motor
