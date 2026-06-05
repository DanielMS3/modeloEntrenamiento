# ==============================================================================
# 0. INSTALACIÓN DE DEPENDENCIAS PARA ENTRENAMIENTO
# ==============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# Asegurar uso de GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Ejecutando entrenamiento en: {device.upper()}")

# ==============================================================================
# 1. CARGA DINÁMICA DE KPIs Y GENERACIÓN DE MAPEOS DESDE EL JSON
# ==============================================================================
RUTA_MATRIZ_CONFIG = "./datos_entrenamiento/indicadores.json"

if not os.path.exists(RUTA_MATRIZ_CONFIG):
    raise FileNotFoundError(f"❌ No se encontró el archivo '{RUTA_MATRIZ_CONFIG}'. Asegúrate de subirlo o verificar la ruta en .")

with open(RUTA_MATRIZ_CONFIG, "r", encoding="utf-8") as f:
    matriz_calidad = json.load(f)

# Extracción ordenada de los atributos de Claro
ATRIBUTOS_IA = [item["atributo"] for item in matriz_calidad]

# Mapeos automáticos Indexados por orden del JSON (0, 1, 2...)
MAPEO_ETIQUETAS = {atributo: idx for idx, atributo in enumerate(ATRIBUTOS_IA)}
MAPEO_INVERSO = {idx: atributo for idx, atributo in enumerate(ATRIBUTOS_IA)}
NUM_LABELS = len(MAPEO_ETIQUETAS)

print(f"✅ Mapeo completado. Atributos detectados para indexar: {NUM_LABELS}")

# ==============================================================================
# 2. CARGA Y PREPARACIÓN DEL DATASET COMERCIAL
# ==============================================================================
RUTA_DATOS_ENTRENAMIENTO = "./datos_entrenamiento/datos_entrenamiento.csv"

if not os.path.exists(RUTA_DATOS_ENTRENAMIENTO):
    raise FileNotFoundError(f"❌ Falta el archivo '{RUTA_DATOS_ENTRENAMIENTO}' con los textos y etiquetas.")

# CARGA DE DATOS
df = pd.read_csv(RUTA_DATOS_ENTRENAMIENTO)
print(f"DEBUG: Columnas encontradas en el CSV: {list(df.columns)}")

# NUEVA LÓGICA MÁS ROBUSTA
columna_etiqueta = None
nombres_posibles = ['label', 'etiqueta', 'atributo', 'target', 'labels'] # Añadimos 'labels'

for col in nombres_posibles:
    if col in df.columns:
        columna_etiqueta = col
        break

if columna_etiqueta is None:
    raise KeyError(f"❌ No se encontró una columna de etiquetas válida. Columnas detectadas: {list(df.columns)}")

print(f"✅ Columna de etiquetas identificada: {columna_etiqueta}")

# Aseguramos que la columna se llame 'labels' para el Trainer de Hugging Face
if columna_etiqueta != 'labels':
    df = df.rename(columns={columna_etiqueta: 'labels'})

# Clonar la columna identificada hacia 'labels' (exigido por Hugging Face)
# Validamos si ya es numérica. Si es texto, aplica el mapeo; si ya es número, lo deja igual.
if df['labels'].dtype == object:
    print("🔄 Detectadas etiquetas en formato Texto. Aplicando mapeo numérico...")
    df['labels'] = df[columna_etiqueta].map(MAPEO_ETIQUETAS)
else:
    print("🔢 Detectadas etiquetas ya indexadas numéricamente de forma nativa.")
    df['labels'] = df[columna_etiqueta].astype(int)

# Limpieza estricta de posibles nulos
df = df.dropna(subset=['texto', 'labels'])
df['labels'] = df['labels'].astype(int)

# Dividir en Entrenamiento (80%) y Validación (20%)
df_train = df.sample(frac=0.8, random_state=42)
df_val = df.drop(df_train.index)

# Convertir a estructura nativa de Hugging Face
dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True))
dataset_val = Dataset.from_pandas(df_val.reset_index(drop=True))

# ==============================================================================
# 3. TOKENIZACIÓN Y CONFIGURACIÓN DEL MODELO BASE (BETO/BERT)
# ==============================================================================
MODEL_NAME = "Recognai/bert-base-spanish-wwm-cased-xnli"

print("📥 Descargando Tokenizer y pesos base del Modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)
model.to(device)

def funcion_tokenizadora(batch):
    # Aseguramos que tome la lista de textos limpia
    return tokenizer(batch["texto"], truncation=True, padding="max_length", max_length=512)

# Tokenizar datos forzando la inclusión y persistencia de las nuevas columnas generadas
print("✨ Ejecutando proceso de tokenización en paralelo...")
dataset_train_tok = dataset_train.map(funcion_tokenizadora, batched=True, remove_columns=["texto"])
dataset_val_tok = dataset_val.map(funcion_tokenizadora, batched=True, remove_columns=["texto"])

# Forzar formato PyTorch Tensor para las columnas que BERT leerá en la GPU
columnas_finales = ["input_ids", "token_type_ids", "attention_mask", "labels"]
dataset_train_tok.set_format(type="torch", columns=columnas_finales)
dataset_val_tok.set_format(type="torch", columns=columnas_finales)

print("✅ Formateo completado con éxito. Columnas preparadas:", dataset_train_tok.column_names)

# ==============================================================================
# 4. MÉTRICAS Y ARCHIVO DE HIPERPARÁMETROS
# ==============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# Ajustamos las rutas al espacio de trabajo garantizado de 
ruta_salida_modelo = "./modelo_qa_entrenado"

training_args = TrainingArguments(
    output_dir="./resultados_checkpoint",
    eval_strategy="epoch",           
    save_strategy="epoch",
    learning_rate=2e-5,              # Tasa de aprendizaje óptima para BERT
    per_device_train_batch_size=4,   # Ajustado para evitar desbordamientos de VRAM en T4
    per_device_eval_batch_size=4,
    num_train_epochs=5,              # Número de pasadas sobre los datos
    weight_decay=0.01,
    load_best_model_at_end=True,     # Guardar el que tenga mejor pérdida en validación
    metric_for_best_model="f1_macro",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_tok,
    eval_dataset=dataset_val_tok,
    compute_metrics=compute_metrics,
)

# ==============================================================================
# 5. EJECUCIÓN Y PERSISTENCIA ABSOLUTA EN DISCO
# ==============================================================================
print("\n🚀 Iniciando Fine-Tuning de BERT en la GPU de ...")
trainer.train()

print(f"\n💾 Forzando guardado de seguridad del modelo entrenado en '{ruta_salida_modelo}'...")
trainer.model.save_pretrained(ruta_salida_modelo)
tokenizer.save_pretrained(ruta_salida_modelo)
print("✅ ¡Estructura de modelo consolidada físicamente en el entorno de  y lista para el Bloque 2!")

