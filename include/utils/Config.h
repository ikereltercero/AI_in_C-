#pragma once
#include <string>
#include <cstdlib>
#include <thread>
#include <iostream>

class Config {
public:
    // Obtiene el puerto (Por defecto: 8080)
    static int get_port() {
        if (const char* env_port = std::getenv("PORT")) {
            return std::stoi(env_port);
        }
        return 8080; 
    }

    // Obtiene la ruta del modelo (Por defecto: "../modelo_v2_iris.bin")
    static std::string get_model_path() {
        
        return "../modelo_v2_iris.bin";
    }

    // Obtiene el número de hilos del ThreadPool (Por defecto: Los núcleos de tu CPU)
    static size_t get_workers() {
        if (const char* env_workers = std::getenv("WORKERS")) {
            return std::stoull(env_workers);
        }
        size_t cores = std::thread::hardware_concurrency();
        return cores > 0 ? cores : 4; 
    }
};