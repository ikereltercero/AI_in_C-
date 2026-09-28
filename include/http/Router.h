#pragma once
#include <string>
#include <unordered_map>
#include <functional>
#include <concepts>
#include "HttpRequest.h"
#include "HttpResponse.h"

// ==============================================================================
// C++20 CONCEPT: Valida que una función sea un Handler HTTP válido en tiempo de compilación.
// Exige que la función acepte (const HttpRequest&, HttpResponse&) y retorne void.
// ==============================================================================
template<typename F>
concept HttpHandlerFunction = requires(F handler, const HttpRequest& req, HttpResponse& res) {
    { handler(req, res) } -> std::same_as<void>;
};

class Router {
private:
    // Estructura clave-valor para asociar "METODO + RUTA" (ej: "GET /predict") a una función lambda o puntero
    std::unordered_map<std::string, std::function<void(const HttpRequest&, HttpResponse&)>> routes;

    std::string make_key(const std::string& method, const std::string& path) const {
        return method + " " + path;
    }

public:
    Router() = default;

    // Método genérico con C++20 Concepts para registrar rutas de forma segura
    template<HttpHandlerFunction F>
    void add_route(const std::string& method, const std::string& path, F&& handler) {
        std::string key = make_key(method, path);
        routes[key] = std::forward<F>(handler);
    }

    // El núcleo del enrutador: busca si la ruta existe y ejecuta su lógica o devuelve un 404
    void route(const HttpRequest& req, HttpResponse& res) const {
        std::string key = make_key(req.method, req.path);
        
        auto it = routes.find(key);
        if (it != routes.end()) {
            // Si la ruta está registrada, ejecutamos su función asociada (Controlador)
            it->second(req, res);
        } else {
            // Si no existe, devolvemos HTTP 404 estandarizado
            res.status_code = 404;
            res.status_message = "Not Found";
            res.headers["Content-Type"] = "text/plain; charset=utf-8";
            res.body = "ERROR 404: La ruta '" + req.path + "' no existe en este servidor.\n";
        }
    }
};