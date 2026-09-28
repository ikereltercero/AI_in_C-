#pragma once
#include "HttpRequest.h"
#include <sstream>
#include <vector>

class HttpParser {
public:
    static HttpRequest parse(const std::string& raw_request) {
        HttpRequest req;
        if (raw_request.empty()) return req;

        std::istringstream stream(raw_request);
        std::string request_line;
        std::getline(stream, request_line);

        // 1. Parsear la Request Line (Ej: GET /predict?f1=5 HTTP/1.1)
        size_t method_end = request_line.find(' ');
        if (method_end != std::string::npos) {
            req.method = request_line.substr(0, method_end);
            
            size_t path_end = request_line.find(' ', method_end + 1);
            if (path_end != std::string::npos) {
                std::string full_path = request_line.substr(method_end + 1, path_end - method_end - 1);
                req.version = request_line.substr(path_end + 1);
                
                // Limpiar retorno de carro (\r) de Windows
                if (!req.version.empty() && req.version.back() == '\r') {
                    req.version.pop_back();
                }

                // Separar la ruta pura de los parámetros
                size_t query_pos = full_path.find('?');
                if (query_pos != std::string::npos) {
                    req.path = full_path.substr(0, query_pos);
                    req.query = full_path.substr(query_pos + 1);
                } else {
                    req.path = full_path;
                }
            }
        }

        // 2. Parsear los Headers (Ej: "Host: localhost:8080")
        std::string header_line;
        while (std::getline(stream, header_line) && header_line != "\r") {
            size_t colon = header_line.find(':');
            if (colon != std::string::npos) {
                std::string key = header_line.substr(0, colon);
                std::string value = header_line.substr(colon + 2); // Saltar ": "
                
                if (!value.empty() && value.back() == '\r') {
                    value.pop_back();
                }
                req.headers[key] = value;
            }
        }

        // 3. Parsear el Body (Para Fase 10 - POST requests)
        std::stringstream body_stream;
        body_stream << stream.rdbuf();
        req.body = body_stream.str();

        return req;
    }
};