#pragma once
#include <string>
#include <unordered_map>

struct HttpRequest {
    std::string_view method;  // "GET", "POST", etc.
    std::string_view path;    // "/predict", "/", "/health"
    std::string_view query;   // "f1=5.1&f2=3.5..."
    std::string_view version; // "HTTP/1.1"

    std::unordered_map<std::string_view, std::string_view> headers;
    std::string_view body;    // Para futuros POSTs con JSON
}; 