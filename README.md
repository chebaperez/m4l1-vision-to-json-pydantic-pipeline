# Demo M4L1: Pipeline Imagen → JSON → Pydantic

Este repositorio contiene el código necesario para la demo y ejercicio en vivo del módulo M4L1: "IA que ve y crea".

## Escenario
Un banco mediano busca automatizar la extracción de datos de formularios de solicitud de crédito escaneados. El objetivo es procesar las imágenes, obtener un JSON estructurado y validarlo contra reglas de negocio usando Pydantic.

## Estructura del Proyecto
- `extract_data_langchain.py`: Script principal que interactúa con la API de OpenAI (GPT-4o).
- `resources/`: Carpeta con ejemplos de prueba y golden_cases.json:
  - `form_clean.png`: Formulario digital, claro y sin errores.
  - `form_handwritten.png`: Formulario con escritura a mano (OCR más complejo).
  - `form_blurred.png`: Formulario con manchas y texto borroso (Edge case).
  - `golden_cases.json`: Dataset para pruebas de benchmark.

## Requisitos
- Python 3.10+
- Una API Key de OpenAI configurada en un archivo `.env`.

## Configuración e Instalación

1. Crear el entorno virtual
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Crear un archivo `.env` en la raíz con tu clave:
   ```env
   OPENAI_API_KEY=tu_clave_aqui
   ```

## Ejecución
Para correr la demo de extracción en una sola imagen (por defecto usa `resources/form_clean.png`):
```bash
python extract_data_langchain.py
```

Para procesar una imagen específica:
```bash
python extract_data_langchain.py --image resources/form_handwritten.png
```

Para ejecutar el benchmark utilizando el dataset de Golden Cases:
```bash
python extract_data_langchain.py --benchmark resources/golden_cases.json
```

--- 

## Pipeline incremental (variaciones de prompts y validaciones) partiendo desde el template

### 1. Simple prompt

```python
model = ChatOpenAI(model=MODEL_NAME)
prompt = (
   "Extrae la información de este formulario y devolvelo en un formato estructurado. "
   "Si algún campo es ilegible, no lo infieras, asignale el valor null. "
)
```
### 2. JSON prompt

```python
model = ChatOpenAI(model=MODEL_NAME)
prompt = (
   "Extrae la información de este formulario y devolvela estrictamente en formato JSON. "
   "Si algún campo es ilegible, no lo infieras, asignale el valor null. "
   "No incluyas explicaciones ni bloques de código markdown, solo el JSON puro. "   
)
```

###  3. JSON prompt + fields list

```python
model = ChatOpenAI(model=MODEL_NAME)
prompt = (
   "Extrae la información de este formulario y devolvela estrictamente en formato JSON. "
   "Usa las siguientes keys: full_name, id_number, loan_amount, application_date, signature_present. "
   "Si algún campo es ilegible, no lo infieras, asignale el valor null. "
   "No incluyas explicaciones ni bloques de código markdown, solo el JSON puro. "
)
```

### 4. Simple prompt + JSON schema with structures output

```python
json_schema = {
   "title": "form_extraction",
   "description": "Información extraída de un formulario bancario",
   "type": "object",
   "properties": {
      "full_name": {"type": "string", "description": "Nombre completo"},
      "id_number": {"type": "string", "description": "Número de documento"},
      "loan_amount": {"type": "number", "description": "Monto solicitado"},
      "application_date": {"type": "string", "description": "Fecha de solicitud"},
      "signature_present": {"type": "boolean", "description": "Si hay firma"}
   },
   "required": ["full_name", "id_number", "loan_amount", "application_date", "signature_present"]
}

model = ChatOpenAI(model=MODEL_NAME).with_structured_output(json_schema)
prompt = (
   "Extrae la información de este formulario. "
   "Si algún campo es ilegible, no lo infieras, asignale el valor null. "
)
```

### 5. Simple prompt + Pydantic

```python
class FormFields(BaseModel):
    full_name: str = Field(description="Nombre completo del solicitante.")
    id_number: str = Field(description="Número de documento (DNI/NIE/ID). Debe contener entre 7 y 10 dígitos.")
    loan_amount: float = Field(description="Monto solicitado en formato numérico.")
    application_date: str = Field(description="Fecha de la solicitud en formato YYYY-MM-DD.")
    signature_present: bool = Field(description="Indica si la firma está presente en el documento.")


model = ChatOpenAI(model=MODEL_NAME).with_structured_output(FormFields)
prompt = (
   "Extrae la información de este formulario. "
   "Si algún campo es ilegible, no lo infieras, asignale el valor null. "
)
```

### 6. JSON prompt + Pydantic with '@field_validator'

```python
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

model = ChatOpenAI(model=MODEL_NAME).with_structured_output(FormFields)
prompt = (
   "Extrae la información de este formulario. "
)
```
