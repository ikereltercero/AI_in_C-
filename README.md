# C++20 Neural Network Engine & Linux HTTP Server

Proyecto personal de desarrollo de sistemas en **C++20 y C11** que combina un motor de redes neuronales implementado desde cero, un servidor HTTP sobre sockets Linux y una capa de abstracción de hardware (HAL) en C.

El objetivo es profundizar en **programación de bajo nivel, gestión de memoria, concurrencia, redes y fundamentos matemáticos del aprendizaje automático**, utilizando principalmente las bibliotecas estándar y las APIs del sistema operativo.

## Características principales

### Motor de redes neuronales
- Implementación desde cero de una red neuronal multicapa (**MLP**).
- Operaciones con tensores y propagación hacia delante (*forward propagation*).
- Retropropagación del error (*backpropagation*).
- Optimizador **Adam**.
- Funciones de activación y pérdida: ReLU, MSE y Cross-Entropy con Softmax.

### Entrenamiento y evaluación
- Carga de datasets en formato CSV.
- División de los datos en conjuntos de entrenamiento y prueba.
- Entrenamiento mediante *mini-batches*.
- Evaluación del rendimiento en tareas de clasificación.

### Servidor HTTP en Linux
- Servidor TCP implementado mediante sockets Linux.
- Gestión de eventos con `epoll` y `eventfd`.
- Sockets no bloqueantes.
- Procesamiento concurrente de peticiones mediante un *thread pool* propio.

### API de inferencia
- Endpoints HTTP para consultar el estado del servidor y obtener predicciones.
- Soporte para parámetros en URL y peticiones JSON.
- Integración del motor de inferencia con el servidor HTTP.

### Capa de abstracción de hardware (HAL)
- Interfaz independiente en C para inicializar el driver, obtener muestras y cerrar el dispositivo.
- Separación entre la interfaz de acceso al hardware y su implementación.
- Generación actual de datos sintéticos para simular lecturas de un sensor.

> **Nota:** La HAL utiliza actualmente datos simulados. Todavía no existe integración con un sensor físico.

### Serialización del modelo
- Almacenamiento y recuperación de pesos y sesgos.
- Formato binario propio con identificación y versión.
- Separación entre entrenamiento e inferencia.

### Paralelismo
- Uso de **OpenMP** para paralelizar determinadas operaciones matriciales y cálculos de gradientes.

### Pruebas automatizadas
- Configuración inicial de pruebas con GoogleTest.
- Integración con CTest para la ejecución de pruebas.

## Tecnologías utilizadas

| Categoría | Tecnologías |
|---|---|
| Lenguajes | C++20, C11 |
| Sistemas y redes | Linux, TCP sockets, `epoll`, `eventfd`, `fcntl`, señales POSIX |
| Concurrencia | `std::thread`, `std::mutex`, `std::condition_variable` |
| Gestión de tareas | ThreadPool propio con cola limitada |
| Machine Learning | MLP, tensores, backpropagation, Adam |
| Funciones matemáticas | ReLU, MSE, Cross-Entropy, Softmax |
| Paralelismo | OpenMP |
| Construcción | CMake |
| Testing | GoogleTest, CTest |
| JSON | nlohmann/json |

## Arquitectura

El proyecto está organizado en componentes independientes para facilitar su mantenimiento y ampliación.

| Componente | Responsabilidad |
|---|---|
| **Core de IA** | Tensores, capas neuronales, modelo MLP, optimizador y cargador de datos. |
| **Servidor HTTP** | Gestión de conexiones TCP, análisis de peticiones HTTP y distribución del trabajo entre hilos. |
| **Router** | Asociación de métodos y rutas HTTP con sus respectivos controladores. |
| **HAL** | Abstracción de la interfaz de acceso al sensor, implementada en C. |
| **Serialización** | Persistencia y recuperación de los parámetros entrenados para su uso en inferencia. |

## Requisitos

- Sistema operativo Linux.
- Compilador compatible con C++20 y C11.
- CMake 3.14 o superior.
- OpenMP.
- Conexión a Internet durante la configuración inicial de CMake para descargar las dependencias declaradas en el proyecto.

## Compilación e instalación

Clona el repositorio:

```bash
git clone https://github.com/ikereltercero/AI_in_C.git
cd AI_in_C
```

Configura el proyecto en modo Release:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Compila utilizando los núcleos disponibles:

```bash
cmake --build build -j"$(nproc)"
```

## Entrenamiento del modelo

Desde el directorio `build`, ejecuta:

```bash
./ia_motor
```

El programa carga el dataset CSV, divide los datos en conjuntos de entrenamiento y prueba, entrena la red neuronal y guarda el modelo resultante en formato binario.

**Importante:** La ruta del dataset debe coincidir con la configuración relativa utilizada por el programa.

## Ejecución del servidor HTTP

Una vez generado el modelo, inicia el servidor desde el directorio `build`:

```bash
./ai_server
```

Por defecto, el servidor escucha en el puerto `8080`.

Puedes configurar el puerto y el número de hilos de trabajo mediante las variables de entorno `PORT` y `WORKERS`:

```bash
PORT=8081 WORKERS=4 ./ai_server
```

## API HTTP

### 1. Estado del servidor

**Endpoint:** `GET /health`

Consulta el estado básico del servicio.

```bash
curl http://localhost:8080/health
```

### 2. Predicción mediante GET

**Endpoint:** `GET /predict`

Envía cuatro características numéricas mediante parámetros URL.

```bash
curl "http://localhost:8080/predict?f1=5.1&f2=3.5&f3=1.4&f4=0.2"
```

El servidor utiliza los valores recibidos como entrada del modelo y devuelve la clase predicha.

### 3. Predicción mediante JSON

**Endpoint:** `POST /api/v1/predict`

Permite enviar las características de entrada en el cuerpo de una petición JSON.

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"f1":5.1,"f2":3.5,"f3":1.4,"f4":0.2}'
```

### 4. Lectura simulada mediante HAL

**Endpoint:** `GET /api/v1/sensor`

```bash
curl http://localhost:8080/api/v1/sensor
```

Genera una muestra sintética mediante la HAL en C y utiliza sus cuatro primeros valores como entrada del modelo de inferencia.

Esta ruta permite probar la integración entre el driver, el servidor HTTP y el motor neuronal sin necesidad de disponer de hardware físico.

## Pruebas

El proyecto incluye una configuración inicial de pruebas automatizadas con GoogleTest y CTest.

Para compilar y ejecutar las pruebas configuradas:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

La cobertura de pruebas está pendiente de ampliación.

## Posibles mejoras

- Ampliar la cobertura de pruebas unitarias y de integración.
- Mejorar la validación de peticiones HTTP y la gestión de lecturas parciales.
- Incorporar normalización de entradas y validación del modelo durante su carga.
- Añadir benchmarks reproducibles de inferencia y concurrencia.
- Sustituir el driver simulado por una implementación conectada a hardware real.

## Objetivo del proyecto

Construir una base práctica para explorar la intersección entre **C++ de sistemas, redes Linux, concurrencia e inferencia de redes neuronales**, manteniendo una arquitectura modular y ampliable.

El proyecto busca combinar los fundamentos del aprendizaje automático con conceptos de ingeniería de sistemas, desde la implementación matemática del modelo hasta su exposición mediante un servicio HTTP concurrente.
