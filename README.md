# **RAG: Un Asistente de Información y Generación de Texto**

## **¿Qué es un RAG?**

Un RAG (Retrieve-Augment-Generate) es un asistente que busca información y la utiliza para generar texto nuevo y coherente. Es una tecnología que combina búsqueda de información y generación de texto para crear contenido nuevo y valioso.

### Aplicaciones de RAG

RAG tiene varias aplicaciones interesantes:

-   **Chatbots y asistentes virtuales**: RAG mejora la capacidad de los chatbots para responder a preguntas complejas y generar texto personalizado.
-   **Generación de contenido**: RAG puede automatizar la creación de contenido, como artículos y resúmenes, ahorrando tiempo y esfuerzo.
-   **Análisis de sentimiento**: RAG analiza el sentimiento y la opinión en texto, lo que es útil en marketing y finanzas.
-   **Investigación y desarrollo**: RAG puede ser utilizado para ayudar a los investigadores a analizar grandes cantidades de datos y generar hipótesis y conclusiones.
-   **Atención al cliente**: RAG puede ser utilizado para generar respuestas personalizadas a preguntas frecuentes y resolver problemas de los clientes de manera más eficiente.
## **Tecnologías a Utilizar**

-   **RAG (Retrieve-Augment-Generate)**: tecnología de análisis de texto y generación de contenido
-   **Python**: lenguaje de programación para implementar el sistema
-   **Langchain**: biblioteca de Python para trabajar con cadenas de texto y análisis de lenguaje natural
-   **Hugging Face**: biblioteca de Python para trabajar con modelos de lenguaje y análisis de texto
-   **OpenAI**: plataforma de inteligencia artificial para utilizar sus herramientas de análisis de texto y respuesta a preguntas
-   **FAISS**: biblioteca de Python para crear un almacenamiento de vectores para búsqueda de texto


## **Ventajas del Proyecto**

-   **Automatización de la generación de contenido**: el sistema puede ahorrar tiempo y esfuerzo en la generación de contenido
-   **Análisis de texto más eficiente**: el sistema puede analizar textos de manera más eficiente y precisa que los métodos tradicionales
-   **Mejora de la atención al cliente**: el sistema puede mejorar la atención al cliente al generar respuestas personalizadas y resolver problemas de manera más eficiente

## **Desafíos del Proyecto**

-   **Complejidad del análisis de texto**: el análisis de texto puede ser un proceso complejo y requiere una gran cantidad de datos y recursos
-   **Calidad de la generación de contenido**: la calidad de la generación de contenido puede variar dependiendo de la calidad de los datos y la complejidad del modelo
-   **Integración con otras tecnologías**: la integración del sistema con otras tecnologías puede ser un desafío y requiere una gran cantidad de trabajo y recursos.
## ETAPA 1: Preparación del Entorno

### Instalación de Herramientas para Análisis de Textos

Preparamos el entorno de ejecución para desempeñar la tarea.

```bash
!pip install pypdf langchain_community langchain-text-splitters langchain_chroma sentence_transformers faiss-cpu openai
		
```


### Cargando un Archivo PDF con Langchain

Éste código carga un archivo PDF y extrae su contenido utilizando la herramienta Langchain, PyPDFLoader.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("libro1.pdf")
pages = loader.load()
print(pages)
		
```


### Unir el Contenido de un Archivo PDF en un Solo Texto

Éste código toma un archivo PDF y une todo su contenido en un solo texto, eliminando los saltos de línea y agregando espacios para separar las páginas.

```python
textoSalida = ''
for Document in pages:
    # Eliminar los saltos de línea y agregar el contenido al textoSalida
    texto_limpio = Document.page_content.replace('\n','')
    textoSalida += texto_limpio +''  # Añadir un espacio para separar los contenidos

print(textoSalida)
		
```


### Dividir un Texto en Pedazos Más Pequeños

Éste código configura una herramienta para dividir un texto en pedazos más pequeños, con un tamaño máximo de 500 caracteres y un solapamiento de 100 caracteres entre cada pedazo.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_text(textoSalida)
print(texts)
		
```


## ETAPA 2: Configuración del Modelo de Lenguaje

### Configurar un Modelo de Lenguaje para Análisis de Texto

Imagina que quieres analizar un texto y entender su significado. Para hacer esto, necesitas un modelo de lenguaje que pueda comprender el texto y extraer información importante de él.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

def embeddingSetup(model):
    embeddings = HuggingFaceEmbeddings(model_name=model)
    print("Inicialización de Embeddings")
    return embeddings

embeddings = embeddingSetup('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
		
```


### Crear un Almacenamiento de Vectores para Búsqueda de Texto

Este código crea un sistema de búsqueda de texto que permite buscar palabras o frases específicas dentro de un archivo de texto de manera rápida y eficiente.

```python
from langchain.vectorstores import FAISS

def vectorStoreCreation(archivo, embedding):
    knowledge_base = FAISS.from_texts(archivo, embedding)
    print("VECTORSTORE CREADA CON EXITO: ")
    return knowledge_base

knowledge_base = vectorStoreCreation(texts, embeddings)
		
```


## ETAPA 3: Configuración de la Conexión con OpenAI

### Configurar la Conexión con OpenAI

Éste código configura la conexión con OpenAI para utilizar sus herramientas de análisis de texto y respuesta a preguntas.

```python
def openAIConfig(model_names, chains):
    os.environ["OPEN
