#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <unordered_map>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <atomic>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/eventfd.h> // Vital para notificaciones ultrarrápidas entre hilos

#include <nlohmann/json.hpp>

#include "ml/MLP.h"
#include "ml/DataLoader.h"
#include "http/HttpRequest.h"
#include "http/HttpParser.h"
#include "http/HttpResponse.h"
#include "http/Router.h"
#include "utils/Config.h"
#include "net/ThreadPool.h"
#include "utils/Logger.h"
#include "hardware/sensor_hal.h" // Driver C de bajo nivel

using json = nlohmann::json;

// ============================================================
// VARIABLES GLOBALES
// ============================================================

volatile std::sig_atomic_t detener_servidor = 0;

int server_fd = -1;
int epoll_fd = -1;
int shutdown_fd = -1; // Descriptor de archivo de eventos para apagado limpio

constexpr int MAX_EVENTS = 64;


#include <unordered_map>
#include <sstream>
#include <stdexcept>

// Analizador de parámetros GET (acepta std::string_view de forma nativa)
std::unordered_map<std::string, std::string> parse_query_string(std::string_view query) {
    std::unordered_map<std::string, std::string> params;
    std::string query_str(query); // Convertimos el view a string para el stringstream
    std::stringstream ss(query_str);
    std::string item;
    while (std::getline(ss, item, '&')) {
        auto delimiter_pos = item.find('=');
        if (delimiter_pos != std::string::npos) {
            std::string key = item.substr(0, delimiter_pos);
            std::string value = item.substr(delimiter_pos + 1);
            params[key] = value;
        }
    }
    return params;
}

// Validador y conversor seguro de string a float
bool parse_float(const std::string& str, float& out_val) {
    try {
        size_t idx = 0;
        out_val = std::stof(str, &idx);
        return idx == str.size(); // Comprueba que toda la cadena era un número válido
    } catch (...) {
        return false;
    }
}

// ============================================================
// MANEJO DE CTRL+C (APAGADO ATÓMICO)
// ============================================================

void handle_sigint(int) {
    detener_servidor = 1;
    uint64_t u = 1;
    if (shutdown_fd != -1) {
        // Escribir en eventfd despierta a epoll_wait inmediatamente (-1 timeout)
        ssize_t bytes_written = write(shutdown_fd, &u, sizeof(uint64_t));
        (void)bytes_written;
    }
}

// ============================================================
// CONFIGURAR UN SOCKET COMO NO BLOQUEANTE
// ============================================================

int set_nonblocking(int sock) {
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) {
        return -1;
    }
    if (fcntl(sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        return -1;
    }
    return 0;
}

// ============================================================
// ENVIO COMPLETO DE DATOS
// ============================================================

bool send_all(int sock, const std::string& data) {
    std::size_t enviados = 0;

    while (enviados < data.size()) {
        ssize_t bytes = send(
            sock,
            data.data() + enviados,
            data.size() - enviados,
            MSG_NOSIGNAL
        );

        if (bytes > 0) {
            enviados += static_cast<std::size_t>(bytes);
            continue;
        }

        if (bytes == -1 && errno == EINTR) {
            continue;
        }

        return false;
    }

    return true;
}

// ============================================================
// LECTURA NO BLOQUEANTE DE PETICIONES HTTP
// ============================================================

std::string leer_peticion_nonblocking(int client_socket) {
    std::string peticion_completa;
    char buffer[1024];

    while (true) {
        ssize_t bytes = recv(client_socket, buffer, sizeof(buffer), 0);

        if (bytes > 0) {
            peticion_completa.append(buffer, bytes);
            
            // Salir si detectamos el final de la cabecera HTTP (\r\n\r\n)
            if (peticion_completa.find("\r\n\r\n") != std::string::npos) {
                break;
            }
        } 
        else if (bytes == 0) {
            break;
        } 
        else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            if (errno == EINTR) {
                continue; 
            }
            break;
        }
    }
    return peticion_completa;
}

// ============================================================
// PROCESAR UNA PETICION CLIENTE EN EL THREADPOOL
// ============================================================

void procesar_cliente(int client_socket, Router& router) {
    try {
        std::string raw_request = leer_peticion_nonblocking(client_socket);

        if (raw_request.empty()) {
            close(client_socket);
            return;
        }

        HttpRequest req = HttpParser::parse(raw_request);
        HttpResponse res;

        router.route(req, res);

        std::string raw_response = res.to_string();

        if (!send_all(client_socket, raw_response)) {
            Logger::log(
                Logger::Level::LOG_INFO,
                "Error enviando respuesta HTTP."
            );
        }
    } catch (const std::exception& e) {
        Logger::log(
            Logger::Level::LOG_INFO,
            std::string("Error procesando cliente: ") + e.what()
        );
    } catch (...) {
        Logger::log(
            Logger::Level::LOG_INFO,
            "Error desconocido procesando cliente."
        );
    }

    close(client_socket);
}

// ============================================================
// MAIN
// ============================================================

int main() {
    // 1. Configurar SIGINT.
    struct sigaction sa {};
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, nullptr) == -1) {
        std::cerr << "[FATAL] Error configurando SIGINT: "
                  << std::strerror(errno) << "\n";
        return 1;
    }

    std::cout << "=== SERVIDOR IA - EPOLL LINUX (C++20 CORE) ===\n";

    Logger::log(
        Logger::Level::LOG_INFO,
        "Iniciando servidor IA con epoll y eventfd."
    );

    // 2. Cargar modelo de IA.
    std::mt19937 gen(1);
    MLP modelo({4, 8, 8, 3}, gen);

    try {
        modelo.load(Config::get_model_path());
        std::cout << "[INFO] Modelo IA cargado.\n";
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Error cargando IA: " << e.what() << "\n";
        return 1;
    }

    // Inicializar el Driver HAL (C puro)
    if (hal_init() != HAL_OK) {
        Logger::log(Logger::Level::LOG_ERROR, "Fallo al inicializar el hardware del sensor HAL.");
        return 1;
    }

    // 3. Configurar router (¡Instanciado antes de añadir rutas!)
    Router router;

    // Ruta 1: Home (GET /)
    router.add_route("GET", "/", [](const HttpRequest&, HttpResponse& res) {
        res.status_code = 200;
        res.status_message = "OK";
        res.headers["Content-Type"] = "text/html; charset=utf-8";
        res.body = "<h1>Motor IA C++20 Operativo</h1>"
                   "<p>Arquitectura desacoplada basada en Routers y Linux epoll.</p>";
    });

    // Ruta 2: Healthcheck (GET /health)
    router.add_route("GET", "/health", [](const HttpRequest&, HttpResponse& res) {
        res.status_code = 200;
        res.status_message = "OK";
        res.headers["Content-Type"] = "application/json";
        res.body = "{\"status\": \"healthy\", \"engine\": \"C++20 Matrix Core\"}";
    });

    // Ruta 3: Inferencia por Query Parameters (GET /predict)
    router.add_route("GET", "/predict", [&modelo](const HttpRequest& req, HttpResponse& res) {
        Logger::log(Logger::Level::LOG_INFO, "Procesando GET /predict (Query mode)");        
        auto params = parse_query_string(req.query);
        
        try {
            auto f1_it = params.find("f1");
            auto f2_it = params.find("f2");
            auto f3_it = params.find("f3");
            auto f4_it = params.find("f4");

            if (f1_it == params.end() || f2_it == params.end() ||
                f3_it == params.end() || f4_it == params.end()) {
                res.status_code = 400;
                res.headers["Content-Type"] = "text/plain";
                res.body = "ERROR: Faltan parametros (f1, f2, f3, f4) en la URL.\n";
                return;
            }

            float f1, f2, f3, f4;
            if (!parse_float(f1_it->second, f1) || !parse_float(f2_it->second, f2) ||
                !parse_float(f3_it->second, f3) || !parse_float(f4_it->second, f4)) {
                res.status_code = 400;
                res.headers["Content-Type"] = "text/plain";
                res.body = "ERROR: Los valores deben ser numericos.\n";
                return;
            }

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
            res.body = "=== INFERENCIA GET ===\nEntrada: [" + std::to_string(f1) + ", " + std::to_string(f2) + ", " + 
                       std::to_string(f3) + ", " + std::to_string(f4) + "]\n"
                       "Especie predicha: " + especie + " (Confianza: " + std::to_string(max_prob) + ")\n";

        } catch (const std::out_of_range&) {
            res.status_code = 400; res.headers["Content-Type"] = "text/plain";
            res.body = "ERROR: Faltan parametros en la URL.\n";
        } catch (const std::invalid_argument&) {
            res.status_code = 400; res.headers["Content-Type"] = "text/plain";
            res.body = "ERROR: Los valores deben ser numericos.\n";
        }
    });

    // Ruta 4: Inferencia IA por JSON (POST /api/v1/predict)
    router.add_route("POST", "/api/v1/predict", [&modelo](const HttpRequest& req, HttpResponse& res) {
        Logger::log(Logger::Level::LOG_INFO, "Procesando POST /api/v1/predict (JSON mode)");    

        auto it = req.headers.find("Content-Type");
        if (it == req.headers.end() || it->second.find("application/json") == std::string::npos) {
            res.status_code = 415; res.body = "{\"error\": \"Requerido application/json\"}";
            return;
        }
        
        try {
            json body_json = json::parse(req.body);

            float f1 = body_json["f1"].get<float>();
            float f2 = body_json["f2"].get<float>();
            float f3 = body_json["f3"].get<float>();
            float f4 = body_json["f4"].get<float>();

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
            res.status_code = 400; res.body = "ERROR 400: Faltan argumentos obligatorios JSON.\n";
        } catch (const std::exception&) {
            res.status_code = 400; res.body = "ERROR 400: JSON malformado o tipos incorrectos.\n";
        }
    });

    // Ruta 5: Lectura de Hardware por Driver C + Inferencia (GET /api/v1/sensor)
    router.add_route("GET", "/api/v1/sensor", [&modelo](const HttpRequest&, HttpResponse& res) {
        sensor_data_t sample;
        
        // 1. Lectura a bajo nivel mediante la capa C (HAL)
        if (hal_read_sample(&sample) != HAL_OK) {
            res.status_code = 500;
            res.body = "{\"error\": \"Fallo de lectura de hardware HAL\"}";
            return;
        }

        // 2. Pasamos los primeros 4 floats de la muestra bruta al modelo de IA (Iris dataset usa 4 features)
        std::vector<float> input_data = {sample.data[0], sample.data[1], sample.data[2], sample.data[3]};
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
        res.headers["Content-Type"] = "application/json";
        res.body = "{\n  \"timestamp_ns\": " + std::to_string(sample.timestamp_ns) + ",\n" +
                   "  \"sample_id\": " + std::to_string(sample.sample_id) + ",\n" +
                   "  \"prediction\": \"" + especie + "\",\n" +
                   "  \"confidence\": " + std::to_string(max_prob) + "\n}";
    });

    // 4. Crear socket TCP principal.
    server_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server_fd == -1) {
        std::cerr << "[FATAL] socket(): " << std::strerror(errno) << "\n";
        return 1;
    }

    // 5. Reutilizar puerto.
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        std::cerr << "[FATAL] setsockopt(): " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    // 6. Activar modo no bloqueante en el socket de escucha.
    if (set_nonblocking(server_fd) == -1) {
        std::cerr << "[FATAL] set_nonblocking(): " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    // 7. Bind y Listen.
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    const int puerto = Config::get_port();
    address.sin_port = htons(static_cast<uint16_t>(puerto));

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1) {
        std::cerr << "[FATAL] bind(): " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, SOMAXCONN) == -1) {
        std::cerr << "[FATAL] listen(): " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    // 8. Crear epoll.
    epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd == -1) {
        std::cerr << "[FATAL] epoll_create1(): " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    // 9. Registrar el socket de escucha en epoll.
    epoll_event ev_server {};
    ev_server.events = EPOLLIN;
    ev_server.data.fd = server_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev_server) == -1) {
        std::cerr << "[FATAL] epoll_ctl (server): " << std::strerror(errno) << "\n";
        close(epoll_fd);
        close(server_fd);
        return 1;
    }

    // 10. Crear e integrar eventfd para apagado limpio (Cero polling desaprovechado)
    shutdown_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (shutdown_fd == -1) {
        std::cerr << "[FATAL] eventfd(): " << std::strerror(errno) << "\n";
        close(epoll_fd);
        close(server_fd);
        return 1;
    }

    epoll_event ev_shutdown {};
    ev_shutdown.events = EPOLLIN;
    ev_shutdown.data.fd = shutdown_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, shutdown_fd, &ev_shutdown) == -1) {
        std::cerr << "[FATAL] epoll_ctl (shutdown): " << std::strerror(errno) << "\n";
        close(shutdown_fd);
        close(epoll_fd);
        close(server_fd);
        return 1;
    }

    // 11. Crear ThreadPool.
    const std::size_t workers = Config::get_workers();
    ThreadPool pool(workers, 100);

    std::cout << "[INFO] Servidor escuchando en http://localhost:" << puerto 
              << " con " << workers << " hilos de procesamiento.\n";

    // 12. Bucle principal de eventos.
    std::vector<epoll_event> events(MAX_EVENTS);

    while (!detener_servidor) {
        // Timeout a -1: El núcleo duerme el hilo exactamente hasta que ocurre un evento real (0% CPU)
        int nfds = epoll_wait(epoll_fd, events.data(), MAX_EVENTS, -1);

        if (nfds == -1) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[ERROR] epoll_wait(): " << std::strerror(errno) << "\n";
            break;
        }

        for (int i = 0; i < nfds; ++i) {
            int current_fd = events[i].data.fd;

            // CASO A: Señal de apagado recibida por eventfd
            if (current_fd == shutdown_fd) {
                detener_servidor = 1;
                break;
            }

            // CASO B: Petición entrante en el socket de escucha principal
            if (current_fd == server_fd) {
                while (!detener_servidor) {
                    sockaddr_in client_addr {};
                    socklen_t client_len = sizeof(client_addr);

                    int client_socket = accept(
                        server_fd,
                        reinterpret_cast<sockaddr*>(&client_addr),
                        &client_len
                    );

                    if (client_socket == -1) {
                        if (errno == EINTR) {
                            continue;
                        }
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break; 
                        }
                        std::cerr << "[ERROR] accept(): " << std::strerror(errno) << "\n";
                        break;
                    }

                    // Configurar socket cliente como no bloqueante
                    if (set_nonblocking(client_socket) == -1) {
                        close(client_socket);
                        continue;
                    }

                    // Encolar cliente en el ThreadPool
                    bool accepted = false;
                    try {
                        accepted = pool.enqueue(
                            [client_socket, &router]() {
                                procesar_cliente(client_socket, router);
                            }
                        );
                    } catch (const std::exception& e) {
                        std::cerr << "[ERROR] Error en ThreadPool: " << e.what() << "\n";
                    }

                    // Si la cola está saturada, responder con HTTP 503
                    if (!accepted) {
                        const std::string overload =
                            "HTTP/1.1 503 Service Unavailable\r\n"
                            "Content-Type: text/plain\r\n"
                            "Connection: close\r\n"
                            "Content-Length: 19\r\n"
                            "\r\n"
                            "Service Unavailable";
                        send_all(client_socket, overload);
                        close(client_socket);
                    }
                }
            }
        }
    }

    // 13. Apagado ordenado de recursos
    Logger::log(Logger::Level::LOG_INFO, "Apagando servidor y liberando descriptores...");

    hal_close(); // Cerrar driver de hardware C

    if (shutdown_fd != -1) {
        close(shutdown_fd);
        shutdown_fd = -1;
    }
    if (server_fd != -1) {
        close(server_fd);
        server_fd = -1;
    }
    if (epoll_fd != -1) {
        close(epoll_fd);
        epoll_fd = -1;
    }

    Logger::log(Logger::Level::LOG_INFO, "Servidor detenido con éxito.");
    std::cout << "[INFO] Servidor apagado limpiamente.\n";

    return 0;
}