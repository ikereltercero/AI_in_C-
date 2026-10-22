#pragma once
#include <string>
#include <unordered_map>
#include <sstream>

struct HttpResponse {
    int status_code = 200;
    std::string status_message = "OK";
    std::unordered_map<std::string, std::string> headers;
    std::string body = "";

    // Constructor por defecto
    HttpResponse() = default;

    // Método mágico: Convierte nuestra estructura limpia en el chorro de bytes 
    // exacto que exige el estándar HTTP/1.1 para viajar por el socket TCP.
    std::string to_string() const {
        std::ostringstream out;
        
        // 1. Request Line
        out << "HTTP/1.1 " << status_code << " " << status_message << "\r\n";
        
        // 2. Headers dinámicos (añadimos Content-Length automáticamente)
        out << "Content-Length: " << body.length() << "\r\n";
        out << "Connection: close\r\n"; // Seguimos sin Keep-Alive por ahora
        
        for (const auto& [key, value] : headers) {
            out << key << ": " << value << "\r\n";
        }
        
        // 3. Separador obligatorio entre Headers y Body (\r\n\r\n)
        out << "\r\n";
        
        // 4. Body
        out << body;
        
        return out.str();
    }
};