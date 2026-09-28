cmake_minimum_required(VERSION 3.14)

project(MotorIA_TFG)

# ==========================================
# 1. Políticas y estándar de C++
# ==========================================
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
include_directories(include)

# ==========================================
# 5. Librería CORE (Motor ML)
# ==========================================
set(MOTOR_SOURCES
    src/autograd.cpp

    src/ml/Layer.cpp
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
)

target_link_libraries(ia_motor
    PRIVATE
        ia_core
        OpenMP::OpenMP_CXX
)

# ==========================================
# 7. Servidor TCP de inferencia
# ==========================================
add_executable(ia_server
    src/main/server_main.cpp
)

target_link_libraries(ia_server
    PRIVATE
        ia_core
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json
)

# Windows: Winsock
if(WIN32)
    target_link_libraries(ia_server
        PRIVATE
            ws2_32
    )
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
