
import os
import re
import uuid
import tempfile

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from pydantic import BaseModel, Field

from pinecone import Pinecone, ServerlessSpec

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 1. ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "ai-menu"
)

PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)

BASE_URL = os.getenv(
    "APP_BASE_URL",
    "http://127.0.0.1:8000"
)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing!")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing!")


# =====================================================
# 2. GLOBAL VARIABLES
# =====================================================

embeddings = None
index = None

# Pinecone Connection
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    api_key=GROQ_API_KEY
)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# =====================================================
# 3. FASTAPI LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global embeddings
    global index

    print("\n===================================")
    print("Starting Restaurant AI Waiter API")
    print("===================================\n")

    # Load HuggingFace Model only once
    print("Loading HuggingFace Embedding Model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding Model Loaded Successfully!")

    # Connect to Pinecone Index
    print("Connecting to Pinecone...")

    if not pc.has_index(INDEX_NAME):

        print("Pinecone Index Not Found. Creating Index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    index = pc.Index(INDEX_NAME)

    print("Pinecone Connected Successfully!")

    print("\nRestaurant AI Waiter API is Ready!")
    print("===================================\n")

    yield

    # Shutdown
    print("\nRestaurant AI Waiter API Stopped!")


# =====================================================
# 4. FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Restaurant AI Waiter API",
    description="Multi Restaurant AI Menu Chatbot",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================
# 5. AI WAITER PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template("""
You are a friendly, professional, and welcoming AI waiter
working at a restaurant.

PERSONALITY:
- Be warm, polite, friendly, and helpful.
- Talk naturally, like a real restaurant waiter.
- Make customers feel comfortable and welcome.
- Keep answers clear, simple, and conversational.
- Avoid robotic or overly formal responses.
- Be respectful and patient.

LANGUAGE:
- Always reply in the same language used by the customer.
- If the customer speaks English, reply in English.
- If the customer speaks Italian, reply in Italian.
- If the customer speaks Bengali, reply in Bengali.

YOUR RESPONSIBILITIES:
1. Answer questions about the restaurant menu.
2. Explain dishes using available menu information.
3. Provide accurate food prices.
4. Recommend dishes only when supported by the menu.
5. Help customers understand available food options.
6. Politely handle greetings and general conversation.

STRICT MENU RULES:
- Use ONLY the provided context for menu-related information.
- Never invent dish names, prices, ingredients, or availability.
- If a dish is not mentioned in the context, politely explain
  that you cannot find it on the current menu.
- Never guess prices or ingredients.
- If information is unclear, say so honestly.
- Do not confuse one dish with another.

CONVERSATION STYLE:
- Keep responses friendly and natural.
- Do not give unnecessarily long answers.
- When appropriate, ask if the customer needs anything else.
- Never pressure customers to order.
- Do not repeat the same greeting in every response.

MENU CONTEXT:
{context}

CUSTOMER MESSAGE:
{question}

YOUR RESPONSE:
""")


# =====================================================
# 6. CHAT REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )


# =====================================================
# 7. TEXT CLEANING FUNCTION
# =====================================================

def clean_documents(documents):

    cleaned_documents = []

    for doc in documents:

        text = doc.page_content

        # Extra spaces remove
        text = re.sub(r"\s+", " ", text)

        # Remove unnecessary spaces
        text = text.strip()

        if text:

            doc.page_content = text

            cleaned_documents.append(doc)

    return cleaned_documents


# =====================================================
# 8. RESTAURANT MENU UPLOAD API
# =====================================================

@app.post("/api/restaurants/upload")
async def upload_restaurant_menu(

    restaurant_name: str = Form(...),

    file: UploadFile = File(...)

):

    if embeddings is None or index is None:

        raise HTTPException(
            status_code=503,
            detail="AI services are not ready."
        )

    restaurant_name = restaurant_name.strip()

    if not restaurant_name:

        raise HTTPException(
            status_code=400,
            detail="Restaurant name is required."
        )

    # Validate PDF extension
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Read PDF
    pdf_bytes = await file.read()

    # Maximum file size 10 MB
    if len(pdf_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="PDF must be smaller than 10 MB."
        )

    # Validate PDF signature
    if not pdf_bytes.startswith(b"%PDF"):

        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    temp_path = None

    try:

        # Generate unique Restaurant ID
        restaurant_id = "rest_" + uuid.uuid4().hex[:12]

        # Create temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(pdf_bytes)

            temp_path = temp_file.name

        # Load PDF
        print(f"Loading PDF: {restaurant_name}")

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found in PDF."
            )

        # Clean PDF text
        cleaned_documents = clean_documents(
            documents
        )

        # Split documents
        chunks = text_splitter.split_documents(
            cleaned_documents
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="No readable menu content found."
            )

        # Add Restaurant Metadata
        for chunk in chunks:

            chunk.metadata["restaurant_id"] = restaurant_id

            chunk.metadata["restaurant_name"] = restaurant_name

        print(f"Creating vectors for {restaurant_name}...")

        # Create separate namespace for each Restaurant
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=restaurant_id
        )

        # Store vectors in Pinecone
        vector_store.add_documents(
            documents=chunks
        )

        print(f"Vectors saved: {restaurant_id}")

        return {

            "success": True,

            "message": "Restaurant menu uploaded successfully!",

            "restaurant_name": restaurant_name,

            "restaurant_id": restaurant_id,

            "total_pages": len(documents),

            "total_chunks": len(chunks),

            "pinecone_index": INDEX_NAME,

            "namespace": restaurant_id,

            "chat_api": f"{BASE_URL}/api/chat/{restaurant_id}"

        }

    except HTTPException:
        raise

    except Exception as e:

        print("Upload Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Menu upload failed: {str(e)}"
        )

    finally:

        # Delete temporary PDF
        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# =====================================================
# 9. RESTAURANT CHAT API
# =====================================================

@app.post("/api/chat/{restaurant_id}")
async def restaurant_chat(

    restaurant_id: str,

    request: ChatRequest

):

    if embeddings is None or index is None:

        raise HTTPException(
            status_code=503,
            detail="AI services are not ready."
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        print(f"Restaurant Chat: {restaurant_id}")

        # Connect to specific Restaurant Namespace
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=restaurant_id
        )

        # Search Restaurant Menu
        docs = vector_store.similarity_search(
            query=question,
            k=3
        )

        if not docs:

            return {

                "success": True,

                "restaurant_id": restaurant_id,

                "question": question,

                "answer": "Sorry, I couldn't find relevant information in this restaurant's menu."

            }

        # Extract Menu Context
        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Create RAG Chain
        rag_chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Generate AI Answer
        answer = rag_chain.invoke({

            "context": context,

            "question": question

        })

        return {

            "success": True,

            "restaurant_id": restaurant_id,

            "question": question,

            "answer": answer

        }

    except Exception as e:

        print("Chat Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


# =====================================================
# 10. HOME API
# =====================================================

@app.get("/")
def home():

    return {

        "success": True,

        "message": "Restaurant AI Waiter API is running!",

        "pinecone_index": INDEX_NAME,

        "status": "online"

    }


# =====================================================
# 11. HEALTH CHECK API
# =====================================================

@app.get("/health")
def health():

    return {

        "api": "running",

        "embedding_model": (
            "loaded" if embeddings is not None else "not loaded"
        ),

        "pinecone": (
            "connected" if index is not None else "not connected"
        )

    }
