#pragma once
#include <iostream>
#include <string>
#include <mutex>
#include <thread>
#include <chrono>
#include <iomanip>

class Logger {
private:
    static inline std::mutex log_mutex; // Mutex estático, compartido por TODAS las instancias/hilos

    // Función privada para obtener la hora actual en formato texto
    static std::string get_time_stamp() {
        const auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
        return std::string(buf);
        //return std::format("{:%Y-%m-%d %H:%M:%S}", now); podria ser sustituido.
    }

public:
    enum class Level { LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_DEBUG };

    static void log(Level lvl, std::string_view message) {
        std::string level_str;
        switch (lvl) {
            case Level::LOG_INFO:    level_str = "[INFO] "; break;
            case Level::LOG_WARNING: level_str = "[WARN] "; break;
            case Level::LOG_ERROR:   level_str = "[ERROR]"; break;
            case Level::LOG_DEBUG:   level_str = "[DEBUG]"; break;
        }

        // BLOQUEO CRÍTICO: Solo un hilo pasa de esta línea a la vez
        std::lock_guard<std::mutex> lock(log_mutex);
        
        std::cout << get_time_stamp() << " | " 
                  << "Thread-" << std::this_thread::get_id() << " | " 
                  << level_str << " " << message << "\n";

        /**
         * mejora: std::osyncstream out(std::cout);

        out << std::format("{:%Y-%m-%d %H:%M:%S}", now)
            << " | Thread-" << std::this_thread::get_id()
            << " | " << level_str
            << " " << message
            << '\n';
         */
    }
};