
# ==============================================================================
# PIPELINE INTEGRAL: AUDITORÍA QA, ANÁLISIS UNIFICADO Y MÓDULO DE SILENCIOS
# ==============================================================================
print("⏳ Cargando entorno optimizado de alta velocidad (Whisper + BERT + Estadísticas)...")
#!pip install openai-whisper pandas matplotlib seaborn soundfile datasets -q

import os
import shutil
import json
import time
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import whisper
import torch
import soundfile as sf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

#%matplotlib inline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Motor configurado en: {device.upper()}")

# ==============================================================================
# 1. LIMPIEZA RADICAL DE ENTORNO Y CONFIGURACIÓN DE RUTAS
# ==============================================================================
RUTA_MATRIZ_CONFIG = "datos_entrenamiento/indicadores.json"
RUTA_MODELO_LOCAL = "modelo_qa_entrenado"
RUTA_CARPETA_AUDIOS = "llamadas"

UMBRAL_APROBACION = 80.0

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS (SIN ELIMINAR CARPETAS)
# ==============================================================================
RUTA_TRANSCRIPCIONES = "transcripciones"
RUTA_REPORTES_INDIVIDUALES = "reportes_qa"
RUTA_METRICAS_SILENCIO = "metricas_silencio"

# Creamos las carpetas solo si no existen, no las borramos para evitar conflictos con OneDrive
for ruta_carpeta in [RUTA_TRANSCRIPCIONES, RUTA_REPORTES_INDIVIDUALES, RUTA_METRICAS_SILENCIO]:
    os.makedirs(ruta_carpeta, exist_ok=True)
    print(f"✅ Carpeta asegurada: {ruta_carpeta}")

with open(RUTA_MATRIZ_CONFIG, "r", encoding="utf-8") as f:
    matriz_calidad = json.load(f)

ATRIBUTOS_IA = [item["atributo"] for item in matriz_calidad]
MAPEO_INVERSO = {idx: atributo for idx, atributo in enumerate(ATRIBUTOS_IA)}
MAPEO_PESOS = {item["atributo"]: float(item["score"].replace("%", "")) for item in matriz_calidad}

UMBRALES_PERSONALIZADOS = {
    "Rapport Personalizado": 0.35,
    "Objeciones Consultivas": 0.25,
    "Lectura de disclaimers": 0.12,
    "Malas practicas": 0.40
}
UMBRAL_BASE_PROPORCIONAL = 0.12

# ==============================================================================
# 2. FUNCIONES RENDIMIENTO VISUAL Y SÍNTESIS DE TEXTO
# ==============================================================================
def graficar_puntos_obtenidos_individual(id_llamada, atributos_evaluados, score_final, estado):
    df_individual = pd.DataFrame(atributos_evaluados)
    df_individual['puntos_obtenidos'] = pd.to_numeric(df_individual['puntos_obtenidos'])
    df_individual = df_individual.sort_values(by='puntos_obtenidos', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 3.2))
    sns.set_theme(style="whitegrid")
    color_titulo = "#2A9D8F" if estado == "APROBADO" else "#E63946"
    color_barras = "#3A86C8" if estado == "APROBADO" else "#E26D5C"
    
    sns.barplot(x='puntos_obtenidos', y='atributo', data=df_individual, color=color_barras, orient='h')
    plt.title(f"Analisis Tecnico: {id_llamada}\nScore Consolidado: {score_final:.2f} / 100 [{estado}]", 
              fontsize=11, fontweight='bold', pad=10, color=color_titulo)
    plt.xlabel("Puntos Asignados")
    plt.ylabel("Atributos")
    plt.tight_layout()
    plt.show()  

def graficar_analisis_unificado_y_silencios(id_llamada, analisis_texto, stats_silencio, estado):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw={'width_ratios': [1.8, 1.2]})
    fig.suptitle(f"Auditoria Integral e Histograma de Silencios - {id_llamada}", fontsize=13, fontweight='bold', y=0.98, color="#1D3557")
    
    # Panel Izquierdo: Diagnóstico de texto unificado
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    fondo_panel = "#E8F5E9" if estado == "APROBADO" else "#FFEBEE"
    borde_panel = "#2E7D32" if estado == "APROBADO" else "#C62828"
    
    ax1.fill_between([0, 10], 0, 10, color=fondo_panel, alpha=0.8)
    ax1.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], color=borde_panel, linewidth=2)
    ax1.text(0.4, 8.8, "DIAGNOSTICO CONSOLIDADO DINAMICO", fontsize=10, fontweight='bold', color="#1D3557")
    
    texto_envuelto = "\n".join(re.findall(r'.{1,70}(?:\s+|$)', analisis_texto))
    ax1.text(0.4, 7.5, texto_envuelto, fontsize=8.5, color="#2B2D42", va='top', family='monospace')

    # Panel Derecho: Métricas de silencio
    ax2.axis('off')
    ax2.fill_between([0, 10], 0, 10, color="#F8F9FA", alpha=0.9)
    ax2.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], color="#6C757D", linewidth=1.5)
    ax2.text(0.5, 8.8, "METRICAS DE SILENCIO (VOZ PASIVA)", fontsize=10, fontweight='bold', color="#495057")
    
    info_silencios = (
        f"  Duracion Audio: {stats_silencio['duracion_total_seg']:.2f}s\n"
        f"  Tiempo en Silencio: {stats_silencio['silencio_total_seg']:.2f}s\n"
        f"  Porcentaje Muerto: {stats_silencio['porcentaje_silencio']:.2f}%\n"
        f"  Cantidad Bloques: {stats_silencio['cantidad_silencios']}\n"
        f"  Silencio mas Largo: {stats_silencio['silencio_max_seg']:.2f}s\n"
        f"  Promedio Silencio: {stats_silencio['silencio_promedio_seg']:.2f}s"
    )
    ax2.text(0.5, 7.0, info_silencios, fontsize=9.5, color="#212529", va='top', family='sans-serif', linespacing=1.6)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.25)
    plt.show()
    print("\n" + "="*95 + "\n")

# ==============================================================================
# 3. CARGA DE MODELOS NATIVOS
# ==============================================================================
print("🎙️ Inicializando modelos fundacionales en memoria...")
tokenizer_entrenado = AutoTokenizer.from_pretrained(RUTA_MODELO_LOCAL)
model_entrenado = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO_LOCAL)

classifier = pipeline(
    "text-classification", model=model_entrenado, tokenizer=tokenizer_entrenado,
    top_k=None, truncation=True, max_length=512, device=0 if device == "cuda" else -1
)
whisper_model = whisper.load_model("small", device=device)

origenes_datos = [
    {"id": os.path.splitext(f)[0], "target": os.path.join(raiz, f)}
    for raiz, _, archivos in os.walk(RUTA_CARPETA_AUDIOS)
    for f in archivos if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg'))
]

total_casos = len(origenes_datos)
todos_los_resultados = []
metricas_globales_silencio = []

# ==============================================================================
# 4. PROCESAMIENTO ATÓMICO CON MÉTRICA DE SILENCIOS INDIVIDUALES
# ==============================================================================
if total_casos > 0:
    print(f"🚀 Iniciando Pipeline sobre {total_casos} audios independientes...\n")

    TAMAÑO_VENTANA = 5           
    SOLAPAMIENTO = 2             

    for idx, caso in enumerate(origenes_datos, 1):
        id_actual = caso["id"]
        origen_puro = caso["target"]
        
        print(f"🔄 [{idx}/{total_casos}] Evaluando de forma aislada: {id_actual}")
        tiempo_inicio = time.time()
        
        sub_frases_maestras = []
        lineas_dialogo_roles = []
        dialogo_estructurado_json = []
        frases_base_agente = []
        ventanas_contexto = []
        mejores_confianzas = {attr: 0.0 for attr in ATRIBUTOS_IA}
        
        try:
            info_audio = sf.info(origen_puro)
            duracion_total_real = info_audio.duration

            res_whisper = whisper_model.transcribe(origen_puro, language="es")
            segmentos = res_whisper.get("segments", [])

            # --- SUB-MÓDULO: CÁLCULO DE SILENCIOS DE ESTA LLAMADA ---
            tiempo_muerto_total = 0.0
            lista_duracion_silencios = []
            ultimo_fin_bloque = 0.0

            for seg in segmentos:
                inicio_seg = float(seg["start"])
                fin_seg = float(seg["end"])
                
                if inicio_seg > ultimo_fin_bloque:
                    brecha = inicio_seg - ultimo_fin_bloque
                    if brecha > 0.4:
                        lista_duracion_silencios.append(brecha)
                        tiempo_muerto_total += brecha
                ultimo_fin_bloque = fin_seg

            if duracion_total_real > ultimo_fin_bloque:
                brecha_final = duracion_total_real - ultimo_fin_bloque
                if brecha_final > 0.4:
                    lista_duracion_silencios.append(brecha_final)
                    tiempo_muerto_total += brecha_final

            stats_silencio_actual = {
                "id_llamada": id_actual,
                "duracion_total_seg": duracion_total_real,
                "silencio_total_seg": tiempo_muerto_total,
                "porcentaje_silencio": round((tiempo_muerto_total / duracion_total_real) * 100, 2) if duracion_total_real > 0 else 0.0,
                "cantidad_silencios": len(lista_duracion_silencios),
                "silencio_max_seg": max(lista_duracion_silencios) if lista_duracion_silencios else 0.0,
                "silencio_promedio_seg": sum(lista_duracion_silencios) / len(lista_duracion_silencios) if lista_duracion_silencios else 0.0
            }
            metricas_globales_silencio.append(stats_silencio_actual)

            # GUARDADO EN CARPETA EXCLUSIVA DE SILENCIOS (INDIVIDUAL)
            with open(os.path.join(RUTA_METRICAS_SILENCIO, f"{id_actual}_silencio.json"), "w", encoding="utf-8") as f_sil:
                json.dump(stats_silencio_actual, f_sil, indent=4, ensure_ascii=False)

            # Estructuración y fragmentación del diálogo
            for seg in segmentos:
                inicio_seg = float(seg["start"])
                fin_seg = float(seg["end"])
                texto_segmento = str(seg["text"]).strip()
                if len(texto_segmento) <= 2: continue
                
                frases_particionadas = [f.strip() for f in re.split(r'(?<=[?.!,])\s+', texto_segmento) if f.strip()]
                num_partes = len(frases_particionadas)
                duracion_estimada = (fin_seg - inicio_seg) / max(1, num_partes)
                
                for i_part, f in enumerate(frases_particionadas):
                    sub_frases_maestras.append({
                        "texto": f, 
                        "tiempo_inicio": round(inicio_seg + (i_part * duracion_estimada), 2), 
                        "tiempo_fin": round(inicio_seg + ((i_part + 1) * duracion_estimada), 2)
                    })

            # Diarización estricta por reglas léxico-temporales
            keywords_agente = ["bienvenido", "asesor", "cédula", "titular", "disclaimer", "validar", "cobertura", "servicio", "mi nombre es", "permítame"]
            for item_frase in sub_frases_maestras:
                frase = item_frase["texto"]
                t_inicio = item_frase["tiempo_inicio"]
                frase_lower = frase.lower()
                
                rol_frase = lineas_dialogo_roles[-1]["rol"] if lineas_dialogo_roles else "AGENTE"
                if t_inicio < 8.0 or any(p in frase_lower for p in keywords_agente):
                    rol_frase = "AGENTE"
                elif frase_lower in ["sí", "no", "correcto", "aló", "bueno", "listo"]:
                    rol_frase = "CLIENTE"
                elif any(p in frase_lower for p in ["yo", "mi cuenta", "no necesito", "me llamaron"]):
                    rol_frase = "CLIENTE"
                
                lineas_dialogo_roles.append({"rol": rol_frase, "texto": frase, "tiempo_inicio": t_inicio, "tiempo_fin": item_frase["tiempo_fin"]})

            # Fusión por bloques conversacionales continuos
            ultimo_rol_registrado = None
            for item in lineas_dialogo_roles:
                if item["rol"] == ultimo_rol_registrado:
                    dialogo_estructurado_json[-1]["texto"] += " " + item["texto"]
                    dialogo_estructurado_json[-1]["tiempo_fin"] = item["tiempo_fin"]
                else:
                    dialogo_estructurado_json.append({
                        "rol": item["rol"], "tiempo_inicio": item["tiempo_inicio"], "tiempo_fin": item["tiempo_fin"], "texto": item["texto"]
                    })
                    ultimo_rol_registrado = item["rol"]

            # Ventanas predictivas para el clasificador BERT
            frases_solo_agente = [b["texto"].strip() for b in dialogo_estructurado_json if b["rol"] == "AGENTE" and len(b["texto"].strip()) > 4]
            for parrafo in frases_solo_agente:
                frases_base_agente.extend([f.strip() for f in re.split(r'[.,;:!?\n]', parrafo) if len(f.strip()) > 4])

            i = 0
            while i < len(frases_base_agente):
                bloque = " ".join(frases_base_agente[i:i + TAMAÑO_VENTANA])
                if len(bloque.strip()) > 15: ventanas_contexto.append(bloque)
                i += (TAMAÑO_VENTANA - SOLAPAMIENTO)

            if not ventanas_contexto: ventanas_contexto = [res_whisper["text"]]

            for salidas_bloque in classifier(ventanas_contexto, batch_size=8):
                for pred in salidas_bloque:
                    match_label = re.search(r'\d+', pred['label'])
                    if not match_label: continue
                    id_label = int(match_label.group())
                    if id_label in MAPEO_INVERSO:
                        attr_detectado = MAPEO_INVERSO[id_label]
                        if pred['score'] > mejores_confianzas[attr_detectado]:
                            mejores_confianzas[attr_detectado] = pred['score']

            # Matriz de Calidad Numérica
            atributos_evaluados_detallado = []
            score_final_llamada = 0.0
            for attr in ATRIBUTOS_IA:
                score_maximo_kpi = MAPEO_PESOS[attr]
                max_confianza = mejores_confianzas[attr]
                umbral_actual = UMBRALES_PERSONALIZADOS.get(attr, UMBRAL_BASE_PROPORCIONAL)

                if attr == "Malas practicas":
                    puntos_obtenidos = 0.0 if max_confianza >= umbral_actual else score_maximo_kpi
                else:
                    if max_confianza >= umbral_actual: puntos_obtenidos = score_maximo_kpi
                    elif max_confianza >= 0.05: puntos_obtenidos = score_maximo_kpi * min(0.85, max(0.35, math.sqrt(max_confianza / umbral_actual)))
                    else: puntos_obtenidos = 0.0

                score_final_llamada += puntos_obtenidos
                atributos_evaluados_detallado.append({
                    "atributo": attr, "score_maximo": score_maximo_kpi,
                    "puntos_obtenidos": round(puntos_obtenidos, 2), "confianza_modelo": round(float(max_confianza), 4)
                })

            score_final_llamada = max(0.0, min(100.0, score_final_llamada))
            estado_final = "APROBADO" if score_final_llamada >= UMBRAL_APROBACION else "REPROBADO"

            # Generación de Análisis Cualitativo Unificado Exclusivo
            bloques_apertura = [b["texto"] for b in dialogo_estructurado_json if b["rol"] == "AGENTE" and b["tiempo_inicio"] <= 15.0]
            saludo_detectado = bloques_apertura[0] if bloques_apertura else (frases_solo_agente[0] if frases_solo_agente else "No estructurado formalmente")
            bloques_cliente = [b for b in dialogo_estructurado_json if b["rol"] == "CLIENTE" and len(b["texto"].strip()) > 5]
            
            if bloques_cliente:
                bloque_cliente_max = max(bloques_cliente, key=lambda x: len(x["texto"]))
                postura_cliente = bloque_cliente_max["texto"]
                tiempo_cliente = bloque_cliente_max["tiempo_inicio"]
                
                bloques_post_agente = [b for b in dialogo_estructurado_json if b["rol"] == "AGENTE" and b["tiempo_inicio"] > tiempo_cliente]
                respuesta_agente = bloques_post_agente[0]["texto"] if bloques_post_agente else "Cierre del canal"
                
                analisis_cualitativo_unico = (
                    f"Llamada auditada bajo el ID {id_actual}. Al inicio, el agente interactúa mediante el discurso: "
                    f"'{saludo_detectado[:65]}...'. Durante el desarrollo de la conversación (segundo {tiempo_cliente}), "
                    f"el usuario interviene con la premisa central: '{postura_cliente[:70]}...', ante la cual la respuesta "
                    f"operativa inmediata registrada por parte del asesor fue: '{respuesta_agente[:65]}...'. "
                    f"El proceso concluye con un score definitivo de {score_final_llamada:.2f} puntos, dictaminando un estado {estado_final} "
                    f"para este registro acústico de forma independiente."
                )
            else:
                analisis_cualitativo_unico = (
                    f"Llamada {id_actual} caracterizada por un flujo conversacional lineal y asistido. El agente inicia con la frase: "
                    f"'{saludo_detectado[:70]}...'. El cliente mantiene una postura cooperativa basada en respuestas cortas de validación. "
                    f"No se detectan objeciones complejas ni desvíos críticos en el canal de voz. Finaliza el proceso con una "
                    f"calificación consolidada de {score_final_llamada:.2f} puntos [{estado_final}]."
                )

            # Guardado en disco estructurado de transcripción y reporte QA
            objeto_transcripcion_final = {
                "id_llamada": id_actual, "dialogo": dialogo_estructurado_json,
                "analisis_llamada": analisis_cualitativo_unico, "metricas_silencio": stats_silencio_actual
            }
            with open(os.path.join(RUTA_TRANSCRIPCIONES, f"{id_actual}_transcripcion.json"), "w", encoding="utf-8") as f_trans:
                json.dump(objeto_transcripcion_final, f_trans, indent=4, ensure_ascii=False)

            reporte_individual_qa = {
                "id_llamada": id_actual, "calificacion_final_QA": round(score_final_llamada, 2), "estado_final_QA": estado_final,
                "atributos_evaluados": atributos_evaluados_detallado, "analisis_llamada": analisis_cualitativo_unico, "metricas_silencio": stats_silencio_actual
            }
            with open(os.path.join(RUTA_REPORTES_INDIVIDUALES, f"{id_actual}_reporte_qa.json"), "w", encoding="utf-8") as f_rep:
                json.dump(reporte_individual_qa, f_rep, indent=4, ensure_ascii=False)

            todos_los_resultados.append(reporte_individual_qa)

            # Despliegue visual por archivo
            print(f"🎯 SCORE OBTENIDO: {score_final_llamada:.2f} / 100 Puntos. [{estado_final}]")
            graficar_puntos_obtenidos_individual(id_actual, atributos_evaluados_detallado, score_final_llamada, estado_final)
            graficar_analisis_unificado_y_silencios(id_actual, analisis_cualitativo_unico, stats_silencio_actual, estado_final)
            
        except Exception as e:
            print(f"⚠️ Error crítico en el elemento {id_actual}: {e}\n")

# ==============================================================================
# 5. MÓDULO ESTADÍSTICO GENERAL DE SILENCIOS (CONSOLIDADO GLOBAL)
# ==============================================================================
if metricas_globales_silencio:
    df_silencios_global = pd.DataFrame(metricas_globales_silencio)
    
    duracion_total_dataset = df_silencios_global["duracion_total_seg"].sum()
    silencio_total_dataset = df_silencios_global["silencio_total_seg"].sum()
    porcentaje_muerto_global = (silencio_total_dataset / duracion_total_dataset) * 100 if duracion_total_dataset > 0 else 0.0
    
    # Construcción de metadatos estadísticos master
    reporte_global_master_silencios = {
        "resumen_ejecutivo": {
            "total_llamadas_procesadas": len(df_silencios_global),
            "tiempo_acumulado_audio_seg": round(float(duracion_total_dataset), 2),
            "tiempo_acumulado_silencio_seg": round(float(silencio_total_dataset), 2),
            "tasa_tiempo_muerto_global_porcentaje": round(float(porcentaje_muerto_global), 2),
            "promedio_silencio_por_llamada_seg": round(float(df_silencios_global['silencio_total_seg'].mean()), 2),
            "silencio_maximo_critico_seg": round(float(df_silencios_global['silencio_max_seg'].max()), 2)
        },
        "detalle_por_llamada": metricas_globales_silencio
    }

    # GUARDADO DEL REPORTE MAESTRO DE SILENCIOS EN SU CARPETA JSON
    with open(os.path.join(RUTA_METRICAS_SILENCIO, "reporte_global_silencios.json"), "w", encoding="utf-8") as f_glob_sil:
        json.dump(reporte_global_master_silencios, f_glob_sil, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print("📊 REPORTE ESTADÍSTICO GLOBAL DE SILENCIOS (DATASET)")
    print("="*50)
    print(f"• Número de llamadas procesadas: {len(df_silencios_global)}")
    print(f"• Tiempo acumulado de audio:     {duracion_total_dataset:.2f} segundos")
    print(f"• Tiempo acumulado en silencio:  {silencio_total_dataset:.2f} segundos")
    print(f"• Tasa de tiempo muerto global:  {porcentaje_muerto_global:.2f}% del tiempo total")
    print(f"• Promedio de silencio por audio:{df_silencios_global['silencio_total_seg'].mean():.2f} segundos")
    print(f"• Silencio más crítico hallado:  {df_silencios_global['silencio_max_seg'].max():.2f} segundos")
    print("="*50 + "\n")
    
    # Gráfica Descriptiva Global
    plt.figure(figsize=(11, 4))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="id_llamada", y="porcentaje_silencio", data=df_silencios_global, palette="Blues_r")
    plt.axhline(df_silencios_global["porcentaje_silencio"].mean(), color="red", linestyle="--", label=f"Media: {df_silencios_global['porcentaje_silencio'].mean():.1f}%")
    plt.title("Distribución Porcentual de Silencio por Identificador de Audio", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("ID de la Llamada")
    plt.ylabel("% Tiempo Muerto / Inactivo")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Guardar reporte unificado master de QA general
    with open("./reportes_qa/reporte_master_qa_audio.json", "w", encoding="utf-8") as f:
        json.dump(todos_los_resultados, f, indent=4, ensure_ascii=False)
        
    print("\n🏁 PIPELINE FINALIZADO. Carpetas reseteadas de forma dinamica y metricas guardadas en JSON.")