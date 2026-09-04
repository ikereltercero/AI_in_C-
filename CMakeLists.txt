cmake_minimum_required(VERSION 3.14)
project(MotorIA_TFG)

# ==========================================
# 1. Políticas y Estándar de C++
# ==========================================
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 2. CONFIGURACIÓN MANUAL DE OPENMP
# ==========================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

# ==========================================
# 3. CREACIÓN DE LA LIBRERÍA CORE (El Motor)
# ==========================================
include_directories(include)

set(MOTOR_SOURCES
    src/autograd.cpp
    src/Neuron.cpp
    src/Layer.cpp
    src/MLP.cpp
    src/Adam.cpp
    src/DataLoader.cpp
)

add_library(ia_core STATIC ${MOTOR_SOURCES})

# ==========================================
# 4. EJECUTABLE PRINCIPAL
# ==========================================
add_executable(ia_motor src/main.cpp)
target_link_libraries(ia_motor PRIVATE ia_core)

# ==========================================
# 5. INTEGRACIÓN DE GOOGLETEST
# ==========================================
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ==========================================
# 6. EJECUTABLE DE PRUEBAS
# ==========================================
add_executable(ia_tests tests/test_autograd.cpp)
target_link_libraries(ia_tests PRIVATE ia_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ia_tests)
