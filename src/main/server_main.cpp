#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <csignal>

#undef ERROR

#include <winsock2.h>
#include <ws2tcpip.h>

#include "ml/MLP.h"
#include "ml/DataLoader.h"
#include "http/HttpRequest.h"
#include "http/HttpParser.h"
#include "http/HttpResponse.h"
#include "http/Router.h"
#include "utils/Config.h" 
#include "net/ThreadPool.h"
#include "utils/Logger.h"


using json = nlohmann::json;

std::atomic<bool> servidor_corriendo{true};
SOCKET server_fd = INVALID_SOCKET;

// Esta función es llamada automáticamente por el Sistema Operativo al pulsar Ctrl+C
void handle_sigint(int) {
    //Logger::log(Logger::WARNING, "\n[SIGINT] Senal de apagado recibida (Ctrl+C). Iniciando Graceful Shutdown...");
    servidor_corriendo = false;
    
    // Cerramos el enchufe principal. Esto forzará a que accept() despierte con un error
    // y permita al hilo principal salir de su bucle infinito.
    if (server_fd != INVALID_SOCKET) {
        closesocket(server_fd);
    }
}
std::string leer_peticion_robusta(SOCKET client_socket) {
    std::string peticion_completa;
    char buffer[1024];
    int headers_end_pos = -1;
    size_t content_length = 0;

    while (true) {
        int bytes = recv(client_socket, buffer, sizeof(buffer), 0);
        if (bytes <= 0) break; // El cliente se desconectó o hubo error

        peticion_completa.append(buffer, bytes);

        // Fase A: Buscar el fin de las cabeceras HTTP
        if (headers_end_pos == -1) {
            size_t pos = peticion_completa.find("\r\n\r\n");
            if (pos != std::string::npos) {
                headers_end_pos = pos + 4; // Inicio del Body

                // Buscar si existe la cabecera Content-Length para saber cuánto Body esperar
                // NOTA: Esto es un mini-parser temporal solo para el tamaño.
                std::string upper_req = peticion_completa; 
                for(auto& c : upper_req) c = toupper(c); // Evitar problemas con "content-length" vs "Content-Length"
                
                size_t cl_pos = upper_req.find("CONTENT-LENGTH:");
                if (cl_pos != std::string::npos) {
                    size_t num_start = cl_pos + 15;
                    size_t num_end = upper_req.find("\r\n", num_start);
                    if (num_end != std::string::npos) {
                        content_length = std::stoull(peticion_completa.substr(num_start, num_end - num_start));
                    }
                }
            }
        }

        // Fase B: Si ya tenemos las cabeceras, verificar si tenemos el Body completo
        if (headers_end_pos != -1) {
            size_t body_recibido = peticion_completa.size() - headers_end_pos;
            if (body_recibido >= content_length) {
                break; // ¡Tenemos la petición HTTP al 100%! Salimos del bucle.
            }
        }
    }
    return peticion_completa;
}

// Función de ayuda para parsear parámetros de consulta (Query String)
std::unordered_map<std::string, std::string> parse_query_string(const std::string& query) {
    std::unordered_map<std::string, std::string> params;
    std::istringstream query_stream(query);
    std::string pair;
    while (std::getline(query_stream, pair, '&')) {
        size_t equal_pos = pair.find('=');
        if (equal_pos != std::string::npos) {
            params[pair.substr(0, equal_pos)] = pair.substr(equal_pos + 1);
        }
    }
    return params;
}

int main() {

    std::signal(SIGINT, handle_sigint);

    std::cout << "=== SERVIDOR IA - ARQUITECTURA MODULAR & C++20 CONCEPTS ===\n";
    Logger::log(Logger::Level::LOG_INFO, "=== INICIANDO MOTOR IA (NIVEL 10 PROTOTIPO) ===");
    // --- 1. CARGA DEL MOTOR DE IA ---
    std::mt19937 gen(1);
    MLP modelo({4, 8, 8, 3}, gen);
    try {
        modelo.load(Config::get_model_path());
        std::cout << "[INFO] Modelo IA cargado exitosamente desde: " << Config::get_model_path() << '\n';
    } catch(const std::exception& e) {
        std::cerr << "[FATAL] Error critico cargando la IA: " << e.what() << "\n";
        return -1;
    }

    // --- 2. CONFIGURACIÓN DEL ROUTER Y ENDPOINTS ---
    Router router;

    // Ruta 1: Home (GET /)
    router.add_route("GET", "/", [](const HttpRequest& req, HttpResponse& res) {
        Logger::log(Logger::Level::LOG_INFO, "Procesando GET /predict (Query param mode)");
        res.status_code = 200;
        res.status_message = "OK";
        res.headers["Content-Type"] = "text/html; charset=utf-8";
        res.body = "<h1>Motor IA C++20 Operativo</h1>"
                   "<p>Arquitectura desacoplada basada en Routers y Concepts.</p>";
    });

    // Ruta 2: Healthcheck (GET /health)
    router.add_route("GET", "/health", [](const HttpRequest& req, HttpResponse& res) {
        Logger::log(Logger::Level::LOG_INFO, "Procesando GET /health");
        res.status_code = 200;
        res.status_message = "OK";
        res.headers["Content-Type"] = "application/json";
        res.body = "{\"status\": \"healthy\", \"engine\": \"C++20 Matrix Core\"}";
    });

    router.add_route("GET", "/predict", [&modelo](const HttpRequest& req, HttpResponse& res) {
    Logger::log(Logger::Level::LOG_INFO, "Procesando GET /predict (Query mode)");        
    auto params = parse_query_string(req.query);
    
    try {
        // En GET usamos std::stof porque todo en la URL es texto
        float f1 = std::stof(params.at("f1"));
        float f2 = std::stof(params.at("f2"));
        float f3 = std::stof(params.at("f3"));
        float f4 = std::stof(params.at("f4"));

        std::vector<float> input_data = {f1, f2, f3, f4};
        BatchView batch = {input_data.data(), nullptr, 1, 4, 3};
        Tensor pred = modelo.inference(batch);

        int ganadora = 0; float max_prob = pred.at(0, 0);
        for (size_t j = 1; j < 3; ++j) {
            if (pred.at(0, j) > max_prob) { max_prob = pred.at(0, j); ganadora = static_cast<int>(j); }
        }
        std::string especie = (ganadora == 0) ? "Setosa" : (ganadora == 1) ? "Versicolor" : "Virginica";

        res.status_code = 200;
        res.status_message = "OK";
        res.headers["Content-Type"] = "text/plain; charset=utf-8";
        res.body = "=== INFERENCIA GET ===\nEntrada: [" + std::to_string(f1) + ", " + std::to_string(f2) + " ...]\n"
                   "Especie predicha: " + especie + " (Confianza: " + std::to_string(max_prob) + ")\n";

    } catch (const std::out_of_range&) {
        res.status_code = 400; res.headers["Content-Type"] = "text/plain";
        res.body = "ERROR: Faltan parametros (f1, f2, f3, f4) en la URL.\n";
    } catch (const std::invalid_argument&) {
        res.status_code = 400; res.headers["Content-Type"] = "text/plain";
        res.body = "ERROR: Los valores deben ser numericos.\n";
    }
    });

        // Ruta 3: Inferencia IA (POST /predict) -> Capturamos el puntero a 'modelo' por referencia
    router.add_route("POST", "/api/v1/predict", [&modelo](const HttpRequest& req, HttpResponse& res) {
        Logger::log(Logger::Level::LOG_INFO, "Procesando POST /api/v1/predict (JSON mode)");    

        auto it = req.headers.find("Content-Type");
        if (it == req.headers.end() || it->second.find("application/json") == std::string::npos) {
            res.status_code = 415; res.body = "{\"error\": \"Requerido application/json\"}";
            return;
        }
        
        try {
            
            json body_json = json::parse(req.body);

            float f1 = (body_json["f1"].get<float>());
            float f2 = (body_json["f2"].get<float>());
            float f3 = (body_json["f3"].get<float>());
            float f4 = (body_json["f4"].get<float>());

            std::vector<float> input_data = {f1, f2, f3, f4};
            BatchView batch = {input_data.data(), nullptr, 1, 4, 3};
            Tensor pred = modelo.inference(batch);

            int ganadora = 0;
            float max_prob = pred.at(0, 0);
            for (size_t j = 1; j < 3; ++j) {
                if (pred.at(0, j) > max_prob) { 
                    max_prob = pred.at(0, j); 
                    ganadora = static_cast<int>(j); 
                }
            }
            std::string especie = (ganadora == 0) ? "Setosa" : (ganadora == 1) ? "Versicolor" : "Virginica";

            res.status_code = 200;
            res.status_message = "OK";
            res.headers["Content-Type"] = "text/plain; charset=utf-8";
            res.body = "=== INFERENCIA NEURONAL EN TIEMPO REAL ===\n"
                       "Entrada analizada: [" + std::to_string(f1) + ", " + std::to_string(f2) + ", " + 
                                              std::to_string(f3) + ", " + std::to_string(f4) + "]\n"
                       "Especie predicha: " + especie + " (Clase " + std::to_string(ganadora) + ")\n"
                       "Confianza de activacion: " + std::to_string(max_prob) + "\n";

        } catch (const std::out_of_range&) {
            res.status_code = 400;
            res.status_message = "Bad Request";
            res.headers["Content-Type"] = "text/plain";
            res.body = "ERROR 400: Faltan argumentos obligatorios en la URL (f1, f2, f3, f4).\n";
        } catch (const std::invalid_argument&) {
            res.status_code = 400;
            res.status_message = "Bad Request";
            res.headers["Content-Type"] = "text/plain";
            res.body = "ERROR 400: Los valores de las caracteristicas deben ser numericos.\n";
        }
    });

    // --- 3. INICIALIZACIÓN DE WINSOCK Y RED TCP ---
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[FATAL] Error al iniciar la capa Winsock.\n";
        return -1;
    }

    server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "[FATAL] No se pudo crear el socket TCP.\n";
        WSACleanup();
        return -1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    int puerto = Config::get_port();
    address.sin_port = htons(puerto);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[FATAL] Error en bind() al puerto " << puerto << ". Puerto ocupado.\n";
        closesocket(server_fd);
        WSACleanup();
        return -1;
    }

    if (listen(server_fd, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "[FATAL] Error en listen() del socket.\n";
        closesocket(server_fd);
        WSACleanup();
        return -1;
    }

    std::cout << "[INFO] Servidor web modular escuchando en http://localhost:8080\n";
    std::cout << "[INFO] Presiona Ctrl+C para detener el servicio.\n\n";
    size_t workers = Config::get_workers();
    ThreadPool pool(workers, 100);
    std::cout << "[INFO] Escuchando en http://localhost:8080 con " << workers << " hilos.\n\n";
    // --- 4. BUCLE DE EVENTOS PRINCIPAL (EVENT LOOP) ---
    while (servidor_corriendo) {
        sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        SOCKET client_socket = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_socket == INVALID_SOCKET) {
            if (!servidor_corriendo) break;
            continue;
        }
        bool accepted = pool.enqueue([client_socket, &router]() {

        DWORD timeout_ms = 5000;
        setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));

        // Lectura de red bufferizada
        std::string raw_request = leer_peticion_robusta(client_socket);
        
        if (raw_request.empty()) {
            closesocket(client_socket);
            return; // Cliente desconectado o error de lectura
        }

        // Procesamiento mediante el Pipeline HTTP

        HttpRequest req = HttpParser::parse(raw_request);
        HttpResponse res;

        Logger::log(Logger::Level::LOG_DEBUG, "Petición recibida: " + req.method + " " + req.path);

        // Delegamos la petición al Router modular que creamos
        router.route(req, res);

        // Serialización y envío de bytes al cliente por TCP
        std::string raw_response = res.to_string();
        send(client_socket, raw_response.c_str(), raw_response.length(), 0);
        closesocket(client_socket);});

        if (!accepted) {
            Logger::log(Logger::Level::LOG_ERROR, "Cola de tareas llena. Rechazando cliente. (HTP 503 Service Unavailable)");
            // Enviamos un error crítico sin pasar por el Router ni parsear nada, 
            // liberando el hilo principal casi al instante.
            std::string overload_response = 
                "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: application/json\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"error\": \"Servidor saturado. Intentalo mas tarde.\"}";
                
            send(client_socket, overload_response.c_str(), overload_response.length(), 0);
            closesocket(client_socket);
        }
    }
    Logger::log(Logger::Level::LOG_INFO, "Hilo principal apagado. Esperando a que el ThreadPool termine sus tareas pendientes...");
    closesocket(server_fd);
    WSACleanup();
    Logger::log(Logger::Level::LOG_INFO, "Servidor apagado de forma segura.");
    return 0;
}