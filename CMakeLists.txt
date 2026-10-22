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
set(CMAKE_C_EXTENSIONS OFF)

# ==========================================
# 2. OpenMP y Threads
# ==========================================
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# ==========================================
# 3. Dependencias externas (FetchContent)
# ==========================================
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ==========================================
# 4. Librería CORE (Motor ML)
# ==========================================
set(MOTOR_SOURCES
    src/autograd.cpp
    src/ml/layer.cpp
    src/ml/MLP.cpp
    src/ml/Adam.cpp
    src/ml/DataLoader.cpp
)

add_library(ia_core STATIC ${MOTOR_SOURCES})

target_include_directories(ia_core 
    PUBLIC 
        include
    PRIVATE 
        .
)

target_link_libraries(ia_core
    PUBLIC
        OpenMP::OpenMP_CXX
)

# ==========================================
# 5. Ejecutables principales (Entrenamiento y Servidor)
# ==========================================
add_executable(ia_motor
    src/main/main.cpp
    src/hardware/sensor_hal.c
)
target_link_libraries(ia_motor PRIVATE ia_core OpenMP::OpenMP_CXX Threads::Threads)

add_executable(ai_server
    src/main/server_main.cpp
    src/hardware/sensor_hal.c
)
target_link_libraries(ai_server
    PRIVATE
        ia_core
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json
        Threads::Threads
)

if(MSVC)
    target_compile_options(ai_server PRIVATE /O2 /W4)
else()
    target_compile_options(ai_server PRIVATE -O3 -Wall -Wextra)
endif()

# ==========================================
# 6. Testing Modulado por Proceso (CTest)
# ==========================================
enable_testing()
include(GoogleTest)

set(TEST_MODULES
    test_tensor
    test_adam
    test_mlp
    test_sensor
)

foreach(test_name ${TEST_MODULES})
    add_executable(${test_name} 
        tests/${test_name}.cpp
        src/hardware/sensor_hal.c
    )

    target_link_libraries(${test_name}
        PRIVATE
            ia_core
            GTest::gtest_main
            Threads::Threads
    )

    gtest_discover_tests(${test_name})
    include(GoogleTest)
endforeach()

