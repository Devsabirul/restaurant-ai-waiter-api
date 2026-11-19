import os
import re
import uuid
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "ai-menu-cohere"
)

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


# ==========================================
# 2. GLOBAL VARIABLES
# ==========================================

embeddings = None
llm = None
pinecone_client = None
pinecone_index = None


# ==========================================
# 3. INITIALIZE APPLICATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global embeddings
    global llm
    global pinecone_client
    global pinecone_index

    print("Starting Restaurant AI Waiter API...")

    # Check API keys

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing!")

    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is missing!")

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing!")

    # Initialize Cohere Embeddings

    print("Loading Cohere Embeddings...")

    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    # Initialize Groq LLM

    print("Loading Groq LLM...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        api_key=GROQ_API_KEY
    )

    # Initialize Pinecone

    print("Connecting to Pinecone...")

    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = [
        item["name"]
        for item in pinecone_client.list_indexes()
    ]

    # Create index if it does not exist

    if PINECONE_INDEX_NAME not in existing_indexes:

        print("Creating Pinecone Index...")

        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    # Wait until index is ready

    import time

    for attempt in range(30):

        index_description = pinecone_client.describe_index(
            PINECONE_INDEX_NAME
        )

        if index_description.status["ready"]:
            break

        print("Waiting for Pinecone Index...")

        time.sleep(2)

    else:
        raise RuntimeError("Pinecone Index is not ready!")

    # Connect to index

    pinecone_index = pinecone_client.Index(
        PINECONE_INDEX_NAME
    )

    # Check embedding dimension

    index_description = pinecone_client.describe_index(
        PINECONE_INDEX_NAME
    )

    if index_description.dimension != 1024:

        raise RuntimeError(
            f"Wrong Pinecone dimension: "
            f"{index_description.dimension}. "
            f"Expected 1024."
        )

    print("Restaurant AI Waiter API is ready!")

    yield

    print("Shutting down API...")


# ==========================================
# 4. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Restaurant AI Waiter API",
    description="AI-powered restaurant menu chatbot API",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 5. CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 6. CHAT REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )


# ==========================================
# 7. TEXT CLEANING FUNCTION
# ==========================================

def clean_documents(documents):

    cleaned_documents = []

    for document in documents:

        text = document.page_content

        # Remove unnecessary whitespace

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        if not text:
            continue

        document.page_content = text

        cleaned_documents.append(document)

    return cleaned_documents


# ==========================================
# 8. RESTAURANT MENU UPLOAD API
# ==========================================

@app.post("/api/restaurants/upload")
async def upload_restaurant_menu(

    restaurant_name: str = Form(...),

    file: UploadFile = File(...)

):

    # Validate restaurant name

    restaurant_name = restaurant_name.strip()

    if not restaurant_name:

        raise HTTPException(
            status_code=400,
            detail="Restaurant name is required."
        )

    # Validate PDF extension

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Read file

    file_content = await file.read()

    # Maximum file size: 10 MB

    if len(file_content) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="PDF size cannot exceed 10 MB."
        )

    # Validate PDF header

    if not file_content.startswith(b"%PDF-"):

        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    temp_file_path = None

    try:

        # Generate unique restaurant ID

        restaurant_id = "rest_" + uuid.uuid4().hex[:12]

        # Save temporary PDF

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_content)

            temp_file_path = temp_file.name

        # ==================================
        # Load PDF
        # ==================================

        loader = PyPDFLoader(temp_file_path)

        documents = loader.load()

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found in PDF."
            )

        total_pages = len(documents)

        # ==================================
        # Clean PDF text
        # ==================================

        documents = clean_documents(documents)

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from PDF."
            )

        # ==================================
        # Split documents
        # ==================================

        text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=50

        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="Could not create menu text chunks."
            )

        # ==================================
        # Add metadata
        # ==================================

        for chunk in chunks:

            chunk.metadata["restaurant_id"] = restaurant_id

            chunk.metadata["restaurant_name"] = restaurant_name

        # ==================================
        # Store in Pinecone
        # ==================================

        vector_store = PineconeVectorStore(

            index=pinecone_index,

            embedding=embeddings,

            namespace=restaurant_id

        )

        vector_store.add_documents(chunks)

        # ==================================
        # Response
        # ==================================

        return {

            "success": True,

            "message": "Restaurant menu uploaded successfully.",

            "restaurant": {

                "restaurant_id": restaurant_id,

                "restaurant_name": restaurant_name

            },

            "menu": {

                "total_pages": total_pages,

                "total_chunks": len(chunks)

            },

            "chat_api": f"/api/chat/{restaurant_id}"

        }

    except HTTPException:

        raise

    except Exception as error:

        print("Upload Error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Menu upload failed. Please try again."
        )

    finally:

        # Remove temporary PDF

        if temp_file_path and os.path.exists(temp_file_path):

            os.remove(temp_file_path)


# ==========================================
# 9. RESTAURANT CHAT PROMPT
# ==========================================

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
4. Recommend dishes only when supported by menu information.
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


# ==========================================
# 10. RESTAURANT CHAT API
# ==========================================

@app.post("/api/chat/{restaurant_id}")
async def restaurant_chat(

    restaurant_id: str,

    request: ChatRequest

):

    try:

        question = request.question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # ==================================
        # Connect to restaurant namespace
        # ==================================

        vector_store = PineconeVectorStore(

            index=pinecone_index,

            embedding=embeddings,

            namespace=restaurant_id

        )

        # ==================================
        # Search relevant menu information
        # ==================================

        relevant_documents = vector_store.similarity_search(

            question,

            k=3

        )

        # ==================================
        # Prepare context
        # ==================================

        context = "\n\n".join(

            document.page_content

            for document in relevant_documents

        )

        # ==================================
        # Generate AI response
        # ==================================

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({

            "context": context,

            "question": question

        })

        return {

            "success": True,

            "restaurant_id": restaurant_id,

            "question": question,

            "answer": answer

        }

    except HTTPException:

        raise

    except Exception as error:

        print("Chat Error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


# ==========================================
# 11. ROOT API
# ==========================================

@app.get("/")
def home():

    return {

        "success": True,

        "message": "Restaurant AI Waiter API is running.",

        "version": "1.0.0"

    }


# ==========================================
# 12. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "service": "Restaurant AI Waiter API"

    }