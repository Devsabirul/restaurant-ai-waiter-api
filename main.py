import os
import re
import uuid
import time
import tempfile
from typing import Annotated, TypedDict
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


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

# Frontend origins: comma-separated URLs
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500"
    ).split(",")
    if origin.strip()
]

# Cookie configuration
COOKIE_SECURE = os.getenv(
    "COOKIE_SECURE",
    "true"
).lower() == "true"

COOKIE_SAMESITE = os.getenv(
    "COOKIE_SAMESITE",
    "none"
).lower()


# ==========================================
# 2. GLOBAL VARIABLES
# ==========================================

embeddings = None
llm = None

pinecone_client = None
pinecone_index = None

chat_graph = None

app_ready = False


# ==========================================
# 3. LANGGRAPH STATE
# ==========================================

class WaiterState(TypedDict):

    messages: Annotated[list, add_messages]

    system_prompt: str


# ==========================================
# 4. INITIALIZE APPLICATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global embeddings
    global llm
    global pinecone_client
    global pinecone_index
    global chat_graph
    global app_ready

    app_ready = False

    print("====================================")
    print("Starting Restaurant AI Waiter API")
    print("====================================")

    # --------------------------------------
    # Check API keys
    # --------------------------------------

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing!")

    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is missing!")

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing!")

    # --------------------------------------
    # Initialize Cohere
    # --------------------------------------

    print("Loading Cohere Embeddings...")

    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    print("Cohere Embeddings initialized!")

    # --------------------------------------
    # Initialize Groq
    # --------------------------------------

    print("Loading Groq LLM...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        api_key=GROQ_API_KEY
    )

    print("Groq LLM initialized!")

    # --------------------------------------
    # Initialize Pinecone
    # --------------------------------------

    print("Connecting to Pinecone...")

    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY
    )

    # Get existing indexes

    existing_indexes = pinecone_client.list_indexes().names()

    print("Existing indexes:", existing_indexes)

    # Create index if not exists

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            f"Index '{PINECONE_INDEX_NAME}' does not exist."
        )

        print("Creating Pinecone index...")

        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

        print("Index creation request submitted!")

    else:

        print(
            f"Index '{PINECONE_INDEX_NAME}' already exists."
        )

    # --------------------------------------
    # Wait for index readiness
    # --------------------------------------

    print("Waiting for Pinecone index readiness...")

    index_description = None

    for attempt in range(90):

        try:

            index_description = pinecone_client.describe_index(
                PINECONE_INDEX_NAME
            )

            if index_description.status["ready"]:

                print("Pinecone control-plane: READY")

                break

            print(
                f"Index is not ready. "
                f"Attempt {attempt + 1}/90"
            )

        except Exception as error:

            print(
                f"Index check failed: {repr(error)}"
            )

        time.sleep(2)

    else:

        raise RuntimeError(
            "Pinecone index did not become ready "
            "within 180 seconds."
        )

    # --------------------------------------
    # Check index dimension
    # --------------------------------------

    if index_description.dimension != 1024:

        raise RuntimeError(
            f"Wrong Pinecone dimension: "
            f"{index_description.dimension}. "
            f"Expected: 1024."
        )

    # --------------------------------------
    # Connect to Pinecone index
    # --------------------------------------

    pinecone_index = pinecone_client.Index(
        PINECONE_INDEX_NAME
    )

    print("Pinecone index object created!")

    # --------------------------------------
    # Test data-plane connection
    # --------------------------------------

    print("Testing Pinecone data-plane connection...")

    for attempt in range(30):

        try:

            stats = pinecone_index.describe_index_stats()

            print("Pinecone connection successful!")

            print("Pinecone index statistics:", stats)

            break

        except Exception as error:

            print(
                f"Data-plane connection failed. "
                f"Attempt {attempt + 1}/30: {repr(error)}"
            )

            time.sleep(2)

    else:

        raise RuntimeError(
            "Pinecone data-plane is not accessible. "
            "Check API key, project and index configuration."
        )

    # ======================================
    # LANGGRAPH MEMORY
    # ======================================

    print("Initializing LangGraph Memory...")

    memory = InMemorySaver()

    def chatbot(state: WaiterState):

        # System prompt is used for this request,
        # but is NOT stored inside conversation messages.

        messages = [
            SystemMessage(content=state["system_prompt"])
        ] + state["messages"]

        response = llm.invoke(messages)

        return {
            "messages": [response]
        }

    builder = StateGraph(WaiterState)

    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "chatbot")

    chat_graph = builder.compile(
        checkpointer=memory
    )

    print("LangGraph Memory initialized!")

    # --------------------------------------
    # Application ready
    # --------------------------------------

    app_ready = True

    print("====================================")
    print("Restaurant AI Waiter API is READY!")
    print("====================================")

    yield

    app_ready = False

    print("Shutting down API...")


# ==========================================
# 5. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Restaurant AI Waiter API",
    description="AI-powered restaurant menu chatbot API",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 6. CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 7. CHAT REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )


# ==========================================
# 8. TEXT CLEANING FUNCTION
# ==========================================

def clean_documents(documents):

    cleaned_documents = []

    for document in documents:

        text = document.page_content

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        if not text:
            continue

        document.page_content = text

        cleaned_documents.append(document)

    return cleaned_documents


# ==========================================
# 9. RESTAURANT MENU UPLOAD API
# ==========================================

@app.post("/api/restaurants/upload")
async def upload_restaurant_menu(

    restaurant_name: str = Form(...),

    file: UploadFile = File(...)

):

    if not app_ready or pinecone_index is None:

        raise HTTPException(
            status_code=503,
            detail="API is not ready. Please try again later."
        )

    restaurant_name = restaurant_name.strip()

    if not restaurant_name:

        raise HTTPException(
            status_code=400,
            detail="Restaurant name is required."
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_content = await file.read()

    if len(file_content) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="PDF size cannot exceed 10 MB."
        )

    if not file_content.startswith(b"%PDF-"):

        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    temp_file_path = None

    try:

        # ----------------------------------
        # Generate restaurant ID
        # ----------------------------------

        restaurant_id = "rest_" + uuid.uuid4().hex[:12]

        print("New restaurant:", restaurant_name)
        print("Restaurant ID:", restaurant_id)

        # ----------------------------------
        # Save temporary PDF
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_content)

            temp_file_path = temp_file.name

        # ----------------------------------
        # Load PDF
        # ----------------------------------

        print("Loading PDF...")

        loader = PyPDFLoader(temp_file_path)

        documents = loader.load()

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found in PDF."
            )

        total_pages = len(documents)

        print("Total PDF pages:", total_pages)

        # ----------------------------------
        # Clean documents
        # ----------------------------------

        documents = clean_documents(documents)

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from PDF."
            )

        # ----------------------------------
        # Split documents
        # ----------------------------------

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

        print("Total chunks:", len(chunks))

        # ----------------------------------
        # Add metadata
        # ----------------------------------

        for chunk in chunks:

            chunk.metadata["restaurant_id"] = restaurant_id

            chunk.metadata["restaurant_name"] = restaurant_name

        # ----------------------------------
        # Store in Pinecone
        # ----------------------------------

        print("Creating Pinecone vector store...")

        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=restaurant_id
        )

        print("Uploading vectors to Pinecone...")

        vector_store.add_documents(chunks)

        print("PDF uploaded successfully!")

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

        print("====================================")
        print("UPLOAD ERROR TYPE:", type(error).__name__)
        print("UPLOAD ERROR DETAILS:", repr(error))
        print("====================================")

        raise HTTPException(
            status_code=500,
            detail="Menu upload failed. Please check server logs."
        )

    finally:

        if temp_file_path and os.path.exists(temp_file_path):

            os.remove(temp_file_path)

        await file.close()


# ==========================================
# 10. GET OR CREATE CUSTOMER SESSION
# ==========================================

@app.post("/api/chat/{restaurant_id}/session")
async def create_chat_session(

    restaurant_id: str,

    request: Request

):

    if not app_ready:

        raise HTTPException(
            status_code=503,
            detail="API is not ready."
        )

    session_id = request.cookies.get(
        "restaurant_chat_session"
    )

    if not session_id:

        session_id = uuid.uuid4().hex

    response = JSONResponse({

        "success": True,

        "restaurant_id": restaurant_id,

        "session_id": session_id,

        "message": "Chat session is ready."

    })

    response.set_cookie(

        key="restaurant_chat_session",

        value=session_id,

        httponly=True,

        secure=COOKIE_SECURE,

        samesite=COOKIE_SAMESITE,

        max_age=86400,

        path="/"

    )

    return response


# ==========================================
# 11. RESTAURANT CHAT API WITH MEMORY
# ==========================================

@app.post("/api/chat/{restaurant_id}")
async def restaurant_chat(

    restaurant_id: str,

    request: ChatRequest,

    http_request: Request

):

    if not app_ready:

        raise HTTPException(
            status_code=503,
            detail="API is not ready."
        )

    try:

        question = request.question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # ----------------------------------
        # Get customer session
        # ----------------------------------

        session_id = http_request.cookies.get(
            "restaurant_chat_session"
        )

        if not session_id:

            raise HTTPException(
                status_code=401,
                detail="Chat session not found. Please start a new chat."
            )

        # ----------------------------------
        # Connect to restaurant namespace
        # ----------------------------------

        vector_store = PineconeVectorStore(

            index=pinecone_index,

            embedding=embeddings,

            namespace=restaurant_id

        )

        # ----------------------------------
        # Search menu information
        # ----------------------------------

        relevant_documents = vector_store.similarity_search(

            question,

            k=3

        )

        # ----------------------------------
        # Prepare context
        # ----------------------------------

        context = "\n\n".join(

            document.page_content

            for document in relevant_documents

        )

        if not context:

            context = "No relevant menu information was found."

        # ----------------------------------
        # System prompt
        # ----------------------------------

        system_prompt = f"""

You are a friendly, professional, and welcoming AI waiter
working at a restaurant.

PERSONALITY:
- Be warm, polite, friendly, and helpful.
- Talk naturally, like a real restaurant waiter.
- Make customers feel comfortable and welcome.
- Keep answers clear, simple, and conversational.
- Avoid robotic or overly formal responses.

LANGUAGE:
- Always reply in the same language used by the customer.
- Support Bengali, English and Italian.

YOUR RESPONSIBILITIES:
1. Answer questions about the restaurant menu.
2. Explain dishes using available menu information.
3. Provide accurate food prices.
4. Recommend dishes only when supported by menu information.
5. Help customers understand available food options.
6. Politely handle greetings and general conversation.

STRICT MENU RULES:
- Use ONLY the provided context for menu-related information.
- Never invent dish names, prices, ingredients or availability.
- If a dish is not mentioned in the context, politely explain
  that you cannot find it on the current menu.
- Never guess prices or ingredients.
- If information is unclear, say so honestly.
- Do not confuse one dish with another.

CONVERSATION STYLE:
- Remember previous conversation.
- Understand follow-up questions.
- Do not repeat the same greeting in every response.
- Keep answers friendly and natural.
- Never pressure customers to order.

MENU CONTEXT:
{context}

"""

        # ----------------------------------
        # LangGraph memory configuration
        # ----------------------------------

        config = {

            "configurable": {

                "thread_id": f"{restaurant_id}_{session_id}"

            }

        }

        # ----------------------------------
        # Generate AI response
        # ----------------------------------

        result = chat_graph.invoke(

            {

                "system_prompt": system_prompt,

                "messages": [

                    HumanMessage(content=question)

                ]

            },

            config=config

        )

        answer = result["messages"][-1].content

        return {

            "success": True,

            "restaurant_id": restaurant_id,

            "session_id": session_id,

            "question": question,

            "answer": answer

        }

    except HTTPException:

        raise

    except Exception as error:

        print("====================================")
        print("CHAT ERROR TYPE:", type(error).__name__)
        print("CHAT ERROR DETAILS:", repr(error))
        print("====================================")

        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please check server logs."
        )


# ==========================================
# 12. ROOT API
# ==========================================

@app.get("/")
def home():

    return {

        "success": True,

        "message": "Restaurant AI Waiter API is running.",

        "version": "1.0.0"

    }


# ==========================================
# 13. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy" if app_ready else "starting",

        "service": "Restaurant AI Waiter API",

        "pinecone_connected": pinecone_index is not None

    }