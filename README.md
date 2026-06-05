---

# Sistema de Auditoría Inteligente QA (IA + Análisis de Audio)

Este proyecto automatiza la auditoría de calidad (QA) de llamadas telefónicas utilizando **Whisper** para la transcripción y un modelo **BERT** entrenado para la clasificación de atributos de calidad.

## 🚀 Estructura del Pipeline

El flujo de trabajo está dividido en dos fases independientes para optimizar el rendimiento y el consumo de recursos:

1. **Fase 1 (`fase1_entrenamiento.py`)**: Entrenamiento del modelo de clasificación y persistencia en la carpeta `modelo_qa_entrenado/`.
2. **Fase 2 (`fase2_analisis.py`)**: Inferencia rápida. Carga el modelo pre-entrenado y analiza audios en tiempo real generando reportes y métricas.

## 🛠 Instalación y Requisitos

### Dependencias

Asegúrate de tener instaladas las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt

```

### Configuración de FFmpeg

El sistema requiere **FFmpeg** para procesar los archivos de audio.

* **Windows**: Descarga [FFmpeg](https://ffmpeg.org/download.html), extrae el ejecutable y añade la ruta de la carpeta `bin` a las variables de entorno (PATH) de tu sistema.

## 📋 Flujo de Uso

### 1. Entrenamiento (Actualización de Inteligencia)

Ejecuta este script solo cuando tengas un nuevo dataset de entrenamiento (`datos_entrenamiento.csv`):

```bash
python fase1_entrenamiento.py

```

### 2. Análisis de Llamadas (Operación diaria)

Coloca tus nuevos audios en `datos_entrenamiento/llamadas/` y ejecuta:

```bash
python fase2_analisis.py

```

## 📂 Organización del Proyecto

* `datos_entrenamiento/`: Contiene el CSV base, los indicadores (JSON) y las llamadas de audio.
* `modelo_qa_entrenado/`: Carpeta del "cerebro" del sistema (modelo y tokenizador).
* `transcripciones/`: Archivos JSON con el texto de cada llamada.
* `reportes_qa/`: Resultados finales de la evaluación y scores.
* `metricas_silencio/`: Análisis de tiempos muertos y actividad en la llamada.

## 📈 Funcionalidades

* **Auditoría basada en IA**: Clasificación automática de comportamientos del agente.
* **Métricas de Silencio**: Cálculo detallado de tiempos muertos para evaluar la fluidez conversacional.
* **Visualización**: Generación de gráficas dinámicas de rendimiento por cada llamada y reportes globales.

---

### Tips de Mantenimiento

> **Importante:** Si realizas cambios en `datos_entrenamiento.csv`, asegúrate siempre de correr la `Fase 1` para actualizar el modelo, de lo contrario, la `Fase 2` seguirá usando el conocimiento antiguo.

¿Necesitas que añada alguna sección especial, como una guía de resolución de problemas comunes o los pasos para instalar algún componente específico adicional?
