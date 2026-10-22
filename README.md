C++20 Neural Network Engine & Linux HTTP Server

Proyecto personal de desarrollo de sistemas en C++20 y C11 que integra un motor de redes neuronales implementado desde cero, un servidor HTTP concurrente sobre sockets Linux y una capa de abstracción de hardware (HAL) desarrollada en C.

El objetivo es profundizar en la programación de bajo nivel, la gestión de memoria, la concurrencia, las redes y los fundamentos matemáticos del aprendizaje automático, utilizando principalmente bibliotecas estándar y APIs del sistema operativo.

Características principales
1. Motor de redes neuronales
Implementación desde cero de una red neuronal multicapa (MLP).
Operaciones con tensores y propagación hacia delante (forward propagation).
Retropropagación del error (backpropagation) para el cálculo de gradientes.
Optimizador Adam para la actualización de parámetros.
Funciones de activación y pérdida: ReLU, MSE y Cross-Entropy con Softmax.
Separación entre entrenamiento e inferencia.
2. Entrenamiento y evaluación
Carga de datasets en formato CSV.
División de datos en conjuntos de entrenamiento y prueba.
Entrenamiento mediante mini-batches.
Evaluación del rendimiento en tareas de clasificación.
Serialización de los parámetros entrenados para su posterior reutilización.
3. Servidor HTTP concurrente
Servidor TCP implementado mediante sockets Linux.
Gestión de eventos con epoll y eventfd.
Uso de sockets no bloqueantes.
Procesamiento concurrente de peticiones mediante un thread pool propio.
Cola de tareas limitada para controlar la distribución del trabajo.
Enrutamiento HTTP mediante controladores asociados a métodos y rutas.
4. API de inferencia
Endpoint para consultar el estado del servidor.
Predicciones mediante parámetros en URL.
Predicciones mediante peticiones HTTP con cuerpos JSON.
Integración directa con el motor de inferencia.
Endpoint para obtener muestras simuladas a través de la HAL.
5. Capa de abstracción de hardware (HAL)
Interfaz independiente implementada en C.
Inicialización del driver, obtención de muestras y cierre del dispositivo.
Separación entre la interfaz de acceso al hardware y su implementación.
Generación de datos sintéticos para simular lecturas de un sensor.
Estado actual: la HAL utiliza datos simulados. El proyecto todavía no dispone de integración con un sensor físico.
6. Serialización del modelo
Almacenamiento y recuperación de pesos y sesgos.
Formato binario propio con identificación y versión.
Persistencia de los parámetros entrenados.
Carga del modelo para realizar inferencias sin necesidad de volver a entrenarlo.
7. Paralelismo
Uso de OpenMP para paralelizar determinadas operaciones matriciales y cálculos de gradientes.
Concurrencia mediante std::thread.
Sincronización con std::mutex y std::condition_variable.
Gestión de tareas mediante un thread pool con cola limitada.
8. Pruebas automatizadas
Configuración de pruebas unitarias con GoogleTest.
Integración con CTest.
Pruebas modulares para tensores, optimizador Adam, modelo MLP y HAL.
Arquitectura

El proyecto se organiza en componentes independientes para facilitar el mantenimiento, las pruebas y la ampliación de funcionalidades.

Componente	Responsabilidad
AI Core	Tensores, capas neuronales, modelo MLP, optimizador y carga de datos.
HTTP Server	Gestión de conexiones TCP, análisis de peticiones HTTP y distribución del trabajo entre hilos.
Router	Asociación de métodos y rutas HTTP con sus respectivos controladores.
HAL	Abstracción de la interfaz de acceso al sensor, implementada en C.
Model Serialization	Persistencia y recuperación de pesos y sesgos para la inferencia.
Testing	Pruebas unitarias y ejecución automatizada mediante GoogleTest y CTest.
Tecnologías utilizadas
Categoría	Tecnologías
Lenguajes	C++20, C11
Sistemas operativos	Linux
Redes	TCP sockets, epoll, eventfd, fcntl, señales POSIX
Concurrencia	std::thread, std::mutex, std::condition_variable
Gestión de tareas	ThreadPool propio con cola limitada
Machine Learning	MLP, tensores, backpropagation, Adam
Funciones matemáticas	ReLU, MSE, Cross-Entropy, Softmax
Paralelismo	OpenMP
Construcción	CMake
Testing	GoogleTest, CTest
JSON	nlohmann/json
Requisitos

Antes de compilar el proyecto, asegúrate de disponer de:

Un sistema operativo Linux.
Un compilador compatible con C++20 y C11.
CMake 3.14 o superior.
OpenMP.
Una conexión a Internet durante la configuración inicial de CMake para descargar las dependencias declaradas en el proyecto.
Compilación e instalación
1. Clonar el repositorio
git clone https://github.com/ikereltercero/AI_in_C.git
cd AI_in_C

2. Configurar el proyecto

Genera el directorio de compilación con la configuración Release:

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

3. Compilar

Compila los ejecutables y las pruebas utilizando los núcleos disponibles:

cmake --build build -j"$(nproc)"


Una vez finalizada la compilación, los binarios estarán disponibles en el directorio build/, según los objetivos definidos en CMake.

Ejecución
1. Motor de entrenamiento (ia_motor)

Carga el dataset CSV, entrena la red neuronal MLP mediante retropropagación y guarda el modelo serializado en disco.

cd build
./ia_motor


Importante: asegúrate de que el dataset se encuentre en la ruta esperada relativa al directorio de ejecución.

Al finalizar el entrenamiento, el modelo generado podrá utilizarse posteriormente para realizar inferencias desde el servidor HTTP, siempre que se haya guardado en la ubicación y con el formato esperados por el servidor.

2. Servidor HTTP (ai_server)

Inicia el servidor TCP, que utiliza epoll para la gestión de eventos y un thread pool para procesar peticiones concurrentes.

cd build
./ai_server


El servidor requiere que el modelo neuronal haya sido entrenado y guardado previamente.

Configuración del puerto y los hilos

Puedes configurar el puerto de escucha y el número de trabajadores mediante variables de entorno:

PORT=8081 WORKERS=4 ./ai_server


Los valores admitidos y los valores predeterminados dependen de la implementación del servidor.

API HTTP

La API permite consultar el estado del servicio, realizar predicciones y obtener muestras simuladas mediante la HAL.

1. Consultar el estado del servidor

Endpoint: GET /health

Permite comprobar el estado básico del servicio.

curl http://localhost:8080/health

2. Realizar una predicción mediante GET

Endpoint: GET /predict

Recibe cuatro características numéricas mediante parámetros URL y devuelve la clase predicha por el modelo.

curl "http://localhost:8080/predict?f1=5.1&f2=3.5&f3=1.4&f4=0.2"

3. Realizar una predicción mediante JSON

Endpoint: POST /api/v1/predict

Permite enviar las características de entrada en el cuerpo de una petición JSON.

curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"f1":5.1,"f2":3.5,"f3":1.4,"f4":0.2}'

4. Obtener una muestra mediante la HAL

Endpoint: GET /api/v1/sensor

curl http://localhost:8080/api/v1/sensor


Esta ruta genera una muestra sintética mediante la HAL implementada en C y utiliza sus cuatro primeros valores como entrada del modelo de inferencia.

Permite probar la integración entre la capa de hardware, el servidor HTTP y el motor neuronal sin necesidad de disponer de un sensor físico.

Nota: las rutas, los parámetros y los ejemplos anteriores describen la interfaz prevista del proyecto. Las respuestas exactas y los códigos HTTP dependen de la implementación actual de cada controlador.
Pruebas automatizadas

El proyecto incorpora una configuración de pruebas unitarias organizada por componentes, con GoogleTest y CTest.

Ejecutar todas las pruebas

Desde la raíz del repositorio:

cmake --build build
ctest --test-dir build --output-on-failure


Para ejecutar las pruebas en paralelo:

ctest --test-dir build --output-on-failure -j"$(nproc)"

Ejecutar pruebas individuales

También puedes ejecutar directamente los binarios de prueba desde el directorio de compilación:

cd build

./test_tensor
./test_adam
./test_mlp
./test_sensor


Los ejecutables permiten aislar los componentes y facilitar la identificación de errores durante el desarrollo.

Objetivos de desarrollo

Este proyecto sirve como entorno práctico para explorar e implementar conceptos de ingeniería de sistemas y aprendizaje automático:

Construcción de un motor neuronal sin depender de frameworks de ML.
Implementación de algoritmos de optimización y propagación de gradientes.
Gestión de memoria y operaciones numéricas en C++.
Programación de servidores de red con APIs de Linux.
Diseño de mecanismos de concurrencia y sincronización.
Separación de responsabilidades mediante una arquitectura modular.
Integración de código C y C++ a través de una HAL.
Persistencia de modelos y separación de las fases de entrenamiento e inferencia.
Automatización de compilación y pruebas.
Estado del proyecto

