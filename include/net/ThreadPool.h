#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <iostream>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks; // La cola de trabajos (clientes)
    
    std::mutex queue_mutex; // Candado para evitar que dos hilos saquen al mismo cliente
    std::condition_variable condition; // Señal para "despertar" a los hilos dormidos
    volatile bool stop; // Bandera para el Graceful Shutdown
    
    size_t max_queue_size;
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency(), size_t max_queue = 1000) : stop(false), max_queue_size(max_queue) {
        // Creamos los hilos trabajadores de antemano
        for(size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        // 1. El hilo coge el candado
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        // 2. Se duerme si no hay tareas y no hemos ordenado parar
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if(this->stop && this->tasks.empty()) return; // Fin del trabajo
                        
                        // 3. Saca la tarea de la cola
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    } // 4. Suelta el candado automáticamente aquí
                    
                    // 5. Ejecuta la tarea (La inferencia HTTP) SIN bloquear a los demás
                    task();
                }
            });
        }
        std::cout << "[INFO] ThreadPool iniciado con " << num_threads << " workers.\n";
    }

    // Método para meter clientes en la cola (Usaremos lambdas de C++20)
    template<class F>
    bool enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            // Pool detenido: no aceptamos nuevas tareas.
            if (stop && tasks.size() >= max_queue_size ) {
                return false;
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one(); // Despierta a 1 hilo dormido
        return true;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all(); // Despierta a todos para que mueran pacíficamente
        for(std::thread &worker: workers) {
            worker.join(); // Espera a que terminen su último trabajo (Graceful Shutdown)
        }
    }
};