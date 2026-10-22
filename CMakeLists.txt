cmake_minimum_required(VERSION 3.14)

project(MotorIA_TFG)

# ==========================================
# 1. Políticas y estándar de C++ y C
# ==========================================
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==========================================
# 2. OpenMP
# ==========================================
find_package(OpenMP REQUIRED)

# ==========================================
# 3. Dependencias externas
# ==========================================
include(FetchContent)

# ------------------------------------------
# nlohmann/json
# ------------------------------------------
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# ------------------------------------------
# GoogleTest
# ------------------------------------------
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ==========================================
# 4. Inclusión de headers
# ==========================================
include_directories(include .)

# ==========================================
# 5. Librería CORE (Motor ML)
# ==========================================
set(MOTOR_SOURCES
    src/autograd.cpp
    src/ml/layer.cpp
    src/ml/MLP.cpp
    src/ml/Adam.cpp
    src/ml/DataLoader.cpp
)

add_library(ia_core STATIC
    ${MOTOR_SOURCES}
)

target_link_libraries(ia_core
    PUBLIC
        OpenMP::OpenMP_CXX
)

# ==========================================
# 6. Ejecutable principal: Entrenamiento
# ==========================================
add_executable(ia_motor
    src/main/main.cpp
    src/hardware/sensor_hal.c
)

target_link_libraries(ia_motor
    PRIVATE
        ia_core
        OpenMP::OpenMP_CXX
)

# ==========================================
# 7. Servidor TCP/epoll de inferencia + HAL
# ==========================================
add_executable(ai_server
    src/main/server_main.cpp          # Tu código epoll + eventfd definitivo
    src/hardware/sensor_hal.c  # Driver C puro integrado
)

target_link_libraries(ai_server
    PRIVATE
        ia_core
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json
        pthread                # Requerido por el ThreadPool y hilos del sistema
)

# Optimización de flags de compilación para rendimiento extremo (-O3)
if(MSVC)
    target_compile_options(ai_server PRIVATE /O2 /W4)
else()
    target_compile_options(ai_server PRIVATE -O3 -Wall -Wextra)
endif()


# ==========================================
# 8. Testing
# ==========================================
enable_testing()

add_executable(ia_tests
    tests/test_tensor.cpp
)

target_link_libraries(ia_tests
    PRIVATE
        ia_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(ia_tests)