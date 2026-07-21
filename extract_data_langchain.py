"""
Live Coding: Pipeline de Extracción Multimodal con LangChain y Pydantic
Este script demuestra cómo extraer información estructurada de imágenes de formularios
utilizando GPT-4o y validación de datos con Pydantic.
"""

import os
import base64
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
from pydantic import ValidationError, BaseModel, Field, field_validator
import re


load_dotenv()
MODEL_NAME = "gpt-4o-mini"


class FormFields(BaseModel):
    full_name: str = Field(description="Nombre completo del solicitante.")
    id_number: str = Field(description="Número de documento (DNI/NIE/ID). Debe contener entre 7 y 10 dígitos.")
    loan_amount: float = Field(description="Monto solicitado en formato numérico.")
    application_date: str = Field(description="Fecha de la solicitud en formato YYYY-MM-DD.")
    signature_present: bool = Field(description="Indica si la firma está presente en el documento.")

    @field_validator('id_number')
    @classmethod
    def validate_id(cls, v: str) -> str:
        # Limpiar espacios y caracteres no numéricos
        clean_id = re.sub(r'\D', '', v)
        if not (7 <= len(clean_id) <= 10):
            raise ValueError("El 'id_number' debe tener entre 7 y 10 dígitos numéricos.")
        return clean_id

    @field_validator('loan_amount')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("El 'loan_amount' debe ser mayor a cero.")
        return v


def encode_image(image_path: str) -> str:
    """Lee un archivo de imagen y lo convierte a base64."""
    print(f"📸 Codificando imagen: {os.path.basename(image_path)}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extraction_pipeline(image_path: str):
    """Ejecuta el pipeline de extracción de datos"""
    
    print(f"\n🚀 Iniciando extracción: {os.path.basename(image_path)}")
    
    # 1. Configurar el modelo con esquema de validación (Pydantic)
    model = ChatOpenAI(model=MODEL_NAME).with_structured_output(FormFields)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra el archivo {image_path}")
        return

    # 2. Definir prompt y encodear imagen
    prompt = (
        "Extrae la información de este formulario. "
    )
    base64_image = encode_image(image_path)

    # 3. Crear mensaje multimodal (Prompt + Imagen)
    print("🧠 Enviando prompt multimodal al LLM...")
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            },
        ]
    )

    # 4. Ejecución y validación
    try:
        parsed_data = model.invoke([message])
        print("✅ Extracción exitosa!")
        print(parsed_data.model_dump_json(indent=2)) # type: ignore
        return parsed_data.model_dump() # type: ignore
    except ValidationError as e:
        print("❌ Error de validación (El modelo devolvió datos que no cumplen con el esquema):")
        print(e)
        return None
    except Exception as e:
        print(f"⚠️ Error durante el procesamiento: {e}")
        return None


def run_benchmark(test_cases_path: str):
    """Ejecuta un benchmark contra un dataset de prueba (Test Cases)"""
    if not os.path.exists(test_cases_path):
        print(f"\n❌ Error: El archivo de benchmark '{test_cases_path}' no existe. Ejecución cancelada.")
        return

    with open(test_cases_path, 'r') as f:
        cases = json.load(f)
    
    results = []
    print(f"\n📊 --- Iniciando Benchmark sobre {len(cases)} casos ---")

    for case in cases:
        print(f"\n=== Evaluando caso: {case['case_id']} | {case['type']} ===")
        
        actual_output = extraction_pipeline(case['image_path'])
        expected_output = case['ground_truth']
        
        # Comparación estricta
        is_correct = actual_output == expected_output
        
        results.append({
            "case_id": case['case_id'],
            "passed": is_correct
        })
        
        if is_correct:
            print(f"✅ Caso '{case['case_id']}' pasó exitosamente.")
        else:
            print(f"❌ Caso '{case['case_id']}' falló la validación estricta.")

    # Resumen
    print(f"\n================================")
    success_rate = sum(1 for r in results if r['passed']) / len(results)
    print(f"\n🏆 Benchmark Finalizado. Exact Match Accuracy: {success_rate * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo de extracción de datos de formularios (Live Coding)."
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default="resources/form_clean.png",
        help="Ruta de la imagen a procesar (ej: resources/form_clean.png)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Ruta al archivo JSON de Test Cases para ejecutar el benchmark"
    )
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args.benchmark)
    else:
        extraction_pipeline(args.image)
