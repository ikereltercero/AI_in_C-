#pragma once
#include <string>
#include <unordered_map>

struct HttpRequest {
    std::string method;  // "GET", "POST", etc.
    std::string path;    // "/predict", "/", "/health"
    std::string query;   // "f1=5.1&f2=3.5..."
    std::string version; // "HTTP/1.1"
    
    std::unordered_map<std::string, std::string> headers;
    std::string body;    // Para futuros POSTs con JSON
}; 