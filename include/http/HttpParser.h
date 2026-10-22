#pragma once
#include "HttpRequest.h"
#include <string_view>

class HttpParser {
public:
    static HttpRequest parse(std::string_view raw_request) {
        HttpRequest req;
        if (raw_request.empty()) return req;

        // 1. Encontrar la frontera entre los Headers y el Body
        size_t body_pos = raw_request.find("\r\n\r\n");
        std::string_view headers_part = raw_request;
        if (body_pos != std::string_view::npos) {
            headers_part = raw_request.substr(0, body_pos);
            req.body = raw_request.substr(body_pos + 4); // El Body es solo una ventana, ¡0 copias!
        }

        // 2. Parsear la Request Line (Ej: GET /predict?f1=5 HTTP/1.1)
        size_t line_end = headers_part.find("\r\n");

        line_end = (line_end == 0) ? headers_part.size() : line_end; // Manejar el caso de una línea vacía
       
        std::string_view request_line = headers_part.substr(0, line_end);
        
        // Deslizamos la ventana general para saltar la primera línea
        if (line_end < headers_part.size()) headers_part.remove_prefix(line_end + 2);

        size_t method_end = request_line.find(' ');
        if (method_end != std::string_view::npos) {
            req.method = request_line.substr(0, method_end);
            
            size_t path_end = request_line.find(' ', method_end + 1);
            if (path_end != std::string_view::npos) {
                std::string_view full_path = request_line.substr(method_end + 1, path_end - method_end - 1);
                req.version = request_line.substr(path_end + 1);
                
                size_t query_pos = full_path.find('?');
                if (query_pos != std::string_view::npos) {
                    req.path = full_path.substr(0, query_pos);
                    req.query = full_path.substr(query_pos + 1);
                } else {
                    req.path = full_path;
                }
            }
        }

        // 3. Parsear los Headers (Bucle de memoria contigua, ¡adiós getline!)
        while (!headers_part.empty()) {
            size_t next_line = headers_part.find("\r\n");
            std::string_view header_line = headers_part.substr(0, next_line);
            
            if (next_line == std::string_view::npos) {
                headers_part = ""; // Fin de los headers
            } else {
                headers_part.remove_prefix(next_line + 2);
            }

            if (header_line.empty()) continue;

            size_t colon = header_line.find(':');
            if (colon != std::string_view::npos) {
                std::string_view key = header_line.substr(0, colon);
                
                // Saltar los espacios en blanco iniciales del valor
                size_t val_start = colon + 1;
                while (val_start < header_line.size() && header_line[val_start] == ' ') {
                    val_start++;
                }
                std::string_view value = header_line.substr(val_start);
                req.headers[key] = value;
            }
        }

        return req;
    }
};