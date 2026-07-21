import os
import base64
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


load_dotenv()
MODEL_NAME = "gpt-4o"


def encode_image(image_path: str) -> str:
    """Lee un archivo de imagen y lo convierte a base64."""
    print(f"📸 Codificando imagen: {os.path.basename(image_path)}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extraction_pipeline(image_path: str):
    """Ejecuta el pipeline de extracción de datos"""
    
    print(f"\n🚀 Iniciando extracción: {os.path.basename(image_path)}")

    # 1. Configurar el modelo
    model = ChatOpenAI(model=MODEL_NAME)
    prompt = (
        "TODO: PROMPT"
    )

    # 2. Encodear imagen
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra el archivo {image_path}")
        return
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
        try:
            print(f"✨ Model Output: {parsed_data.content}")  # type: ignore
            print(type(parsed_data.content))  # type: ignore
        except AttributeError:
            print(f"⚙️ Structured Output: {parsed_data}")
            print({type(parsed_data)})
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")


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
    args = parser.parse_args()
    
    extraction_pipeline(args.image)
