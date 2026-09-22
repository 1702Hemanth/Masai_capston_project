# Zepto Support Assistant

## 1. Overview

The Support Assistant is a small Retrieval-Augmented Generation (RAG)
service built for Zepto customer-support policy questions.

The application uses:

- 8 Zepto policy documents
- Sentence Transformers
- `all-MiniLM-L6-v2` for local embeddings
- ChromaDB for vector storage and retrieval
- LangGraph for intent routing
- Pydantic for structured output validation
- FastAPI for the REST API
- Uvicorn for local execution

The required graded path uses deterministic offline mock logic through
`MOCK_LLM`. No LLM API key or paid service is required.

---

## 2. Module Objective

This module implements a complete RAG pipeline:

1. Ingest the Zepto policy documents.
2. Chunk the documents.
3. Generate local embeddings.
4. Store embeddings in ChromaDB.
5. Classify incoming questions.
6. Retrieve relevant policy context when required.
7. Generate a deterministic mock response.
8. Validate the response using Pydantic.
9. Expose the system through a FastAPI `/ask` endpoint.
10. Provide a Dockerfile for local containerization.

---

## 3. Project Structure

text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── rag.py
├── main.py
├── Dockerfile
├── requirements.txt
├── __init__.py
└── README.md


## 4. Policy Corpus

The application contains eight policy documents.
Document	Topic
doc_01.txt	Delivery Policy
doc_02.txt	Returns & Refunds
doc_03.txt	Membership Tiers
doc_04.txt	Order Tracking
doc_05.txt	Order Cancellation Policy
doc_06.txt	Damaged or Missing Items
doc_07.txt	Gift Cards
doc_08.txt	Customer Support Hours


Each document is stored as a separate text file under docs/.

## 5. Embedding Model

The project uses the open-source Sentence Transformers model:
all-MiniLM-L6-v2
The model runs locally.
No LLM API key is required for the graded baseline.
The embeddings are normalized before storage:
model.encode(
    documents,
    normalize_embeddings=True
)
Because the embeddings are normalized, cosine similarity can be calculated
from the corresponding ChromaDB cosine distance.

## 6. Ingestion Pipeline

The ingestion process is implemented in ingest.py.
The pipeline is:
Policy Documents
       |
       v
Load .txt files
       |
       v
One chunk per document
       |
       v
all-MiniLM-L6-v2
       |
       v
Normalized embeddings
       |
       v
ChromaDB
       |
       v
zepto_policies collection
Because the eight documents are short, one chunk per document is sufficient
for this assignment.
The ChromaDB collection is:
zepto_policies
The collection stores:
- Document ID
- Document text
- Embedding vector
- Source metadata
- Chunk ID

## 7. Running Ingestion

From the project root:
python support_assistant\ingest.py
Successful execution produced:
Loading policy documents...
Documents loaded: 8
Loading embedding model: all-MiniLM-L6-v2
Generating embeddings...
Connecting to ChromaDB...
ChromaDB collection: zepto_policies
Documents indexed: 8

Verification query:
  doc_01: distance=0.0000
  doc_08: distance=0.8096
  doc_02: distance=0.8574

Ingestion completed successfully.
This verifies that all eight corpus documents were indexed.

## 8. Structured Prompt Template

The structured prompt is defined in rag.py.
It follows the required:
Role
Context
Task
Format
Length
skeleton.
The prompt also contains an explicit negative constraint and a few-shot
example.
Role
The assistant is defined as a Zepto customer-support assistant.
Context
The prompt instructs the system to use only the retrieved Zepto policy
information.
Task
The assistant must answer the customer's question using the supplied
context.
Format
The expected response is a JSON object containing:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}
Length
The response should remain concise and directly relevant.
Negative Constraint
The prompt explicitly instructs the assistant:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, refunds, or procedures.
Few-Shot Example
The prompt includes an example question about delivery fees, the relevant
context from doc_01, and a correctly formatted JSON answer.
The structured prompt is primarily used by the optional real-LLM extension.
The graded mock mode does not make an LLM call.

## 9. MOCK_LLM Configuration

The graded baseline is completely offline.
The application checks:
MOCK_LLM
If the variable is unset or set to:
MOCK_LLM=1
the deterministic mock path is used.
No LLM API call is made.
The optional real-LLM extension is activated only when:
MOCK_LLM=0
The required submission does not depend on this optional path.

## 10. LangGraph Architecture

The LangGraph workflow is implemented in rag.py.
The graph uses a TypedDict state and contains three required nodes:
classify_intent
retrieve_and_answer
direct_answer
The graph is:
                    START
                      |
                      v
              +---------------+
              | classify_intent|
              +---------------+
                 /           \
                /             \
       policy_question     general_question
             |                   |
             v                   v
 +-----------------------+  +-------------+
 | retrieve_and_answer   |  | direct_answer|
 +-----------------------+  +-------------+
             |                   |
             v                   v
            END                 END

## 11. Intent Classification

The classify_intent node determines whether retrieval is required.
In mock mode, the following keywords are checked:
delivery
return
refund
membership
tracking
cancel
gift card
support hours
If the lowercased query contains any of these keywords:
policy_question
is returned.
Otherwise:
general_question
is returned.
This classification does not require an LLM.

## 12. Retrieval

The retrieve_and_answer node performs retrieval for policy questions.
The query is embedded using:
all-MiniLM-L6-v2
The embedding is then queried against:
zepto_policies
in ChromaDB.
The top three most similar chunks are retrieved.
The retrieval step always runs for policy questions, regardless of the
MOCK_LLM setting.
The mock mode only changes the final answer-generation step.

## 13. Mock Retrieval Answer

In mock mode, the retrieved context is used directly to build a deterministic
answer.
The response follows the required form:
Based on the retrieved context: <top chunk snippet>
The snippet is taken from the most similar retrieved document.
The response also contains the retrieved document IDs as sources.

## 14. Direct Answer

The direct_answer node handles general questions.
In mock mode, it returns:
I can only answer questions about Zepto policies right now.
No retrieval is performed for this route.
No LLM call is made.
The response contains:
{
  "sources": [],
  "confidence": 1.0
}

## 15. Conditional Routing

The graph contains a conditional edge after classify_intent.
The routing function checks the intent:
policy_question
        |
        v
retrieve_and_answer
or:
general_question
        |
        v
direct_answer
The routing logic itself does not depend on MOCK_LLM.
Only the generation step inside the relevant nodes is affected by the
toggle.

## 16. Pydantic Response Schema

The final response is validated using:
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
The confidence value is constrained to:
0.0 <= confidence <= 1.0
In mock mode:
policy question:
sources = retrieved document IDs
confidence = 1.0

general question:
sources = []
confidence = 1.0
This provides deterministic structured output.

## 17. FastAPI Application

The FastAPI application is implemented in:
main.py
The API provides:
POST /ask
The request schema is:
{
  "query": "string"
}
The response schema is:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}

## 18. Running the FastAPI Application

From the project root:
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
The application runs at:
http://127.0.0.1:7860
The root endpoint returns:
{
  "service": "Zepto Support Assistant",
  "status": "running",
  "mock_llm": true,
  "endpoint": "POST /ask"
}

## 19. Example API Call 1 — Policy Question

Request:
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the delivery fee for orders below INR 149?"}' | ConvertTo-Json -Depth 5
Raw JSON response:
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_07"
  ],
  "confidence": 1.0
}
The query was classified as:
policy_question
and routed to:
retrieve_and_answer
doc_01 was the top retrieved source.

## 20. Example API Call 2 — General Question

Request:
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of India?"}' | ConvertTo-Json -Depth 5
Raw JSON response:
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
The query was classified as:
general_question
and routed to:
direct_answer
No retrieval was performed.

## 21. End-to-End Architecture

The complete pipeline is:
                   INGESTION
                       |
                       v
              docs/*.txt files
                       |
                       v
                 ingest.py
                       |
                       v
              Chunk documents
                       |
                       v
              Sentence Transformer
             all-MiniLM-L6-v2
                       |
                       v
                    ChromaDB
              zepto_policies
                       |
                       |
                       | Query
                       v
                   FASTAPI
                   POST /ask
                       |
                       v
               LangGraph StateGraph
                       |
                       v
               classify_intent
                  /          \
                 /            \
                v              v
       policy_question    general_question
                |              |
                v              v
       retrieve_and_answer  direct_answer
                |
                v
          ChromaDB top-3
                |
                v
        Retrieved context
                |
                v
          Answer generation
                |
                v
        Pydantic validation
                |
                v
              JSON
Stage responsibilities
Stage	Component
Ingestion	ingest.py
Chunking	load_documents() in ingest.py
Embedding	Sentence Transformers all-MiniLM-L6-v2
Vector storage	ChromaDB zepto_policies
Intent routing	classify_intent in rag.py
Retrieval	retrieve_documents() in rag.py
Policy answer	retrieve_and_answer
General answer	direct_answer
Structured validation	SupportResponse Pydantic model
API	main.py FastAPI
Server	Uvicorn


## 22. MOCK_LLM Data Flow

Default Mock Mode
MOCK_LLM unset / MOCK_LLM=1
              |
              v
     Deterministic heuristic
              |
       +------+------+
       |             |
       v             v
   Retrieval       Direct
       |             |
       v             v
  Mock answer    Fixed answer
       |             |
       +------+------+
              |
              v
       Pydantic JSON
No LLM provider is contacted.
Optional Real-LLM Mode
MOCK_LLM=0
     |
     v
Intent generation can use an LLM
     |
     v
Policy queries still use ChromaDB retrieval
     |
     v
Retrieved context + structured prompt
     |
     v
Real LLM answer generation
     |
     v
Pydantic validation
The optional real-LLM path is not required for the graded baseline.

## 23. Docker

A Dockerfile is included at:

support_assistant/Dockerfile

The container uses Python 3.12 and starts the FastAPI application with:

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]

### Build Image

From the project root:

docker build -t zepto-support-assistant ./support_assistant

The image was successfully built locally as:

zepto-support-assistant:latest

### Run Container

powershell
docker run --rm -p 7860:7860 zepto-support-assistant
The API is then available at:
http://127.0.0.1:7860
Docker Verification
The container was successfully started and the FastAPI service responded to
API requests.
The following endpoints were verified:
GET /
POST /ask
The container was also tested with both a Zepto policy question and a general
question. The expected deterministic MOCK_LLM responses were returned.
The Docker image performs ingestion during the image build, so the ChromaDB
policy index is available inside the container when it starts.

### Also update Section 27

Your current checklist says:

> Docker build/run — Requires Docker installation :chatgpt-content-reference{index="2"}

Change:

text
Dockerfile              Complete
Docker build/run        Complete
Architecture documentation  Complete

### Optional Real-LLM Mode

The application reserves `MOCK_LLM=0` for an optional real-LLM provider
integration. The graded submission does not depend on this mode and no
external API key is required or included in the repository.

The fully implemented and tested submission path is the deterministic
offline mode:

MOCK_LLM unset or MOCK_LLM=1

## 24. Dependencies

The module uses:
fastapi
uvicorn
pydantic
langgraph
chromadb
sentence-transformers
Install them with:
python -m pip install -r support_assistant\requirements.txt

## 25. Complete Local Setup

From the project root:
Step 1 — Activate virtual environment
.venv\Scripts\Activate.ps1
Step 2 — Install dependencies
python -m pip install -r support_assistant\requirements.txt
Step 3 — Build ChromaDB index
python support_assistant\ingest.py
Step 4 — Start API
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
Step 5 — Test /ask
Use the two example requests documented above.

## 26. Files Generated

During execution, the following local data is generated:
support_assistant/
└── chroma_db/
The ChromaDB directory contains the locally persisted vector database.
Python may also create:
__pycache__/
These generated Python cache files should not be committed to Git.

## 27. Acceptance Checklist

Requirement	Status
8 policy documents	Complete
Local MiniLM embeddings	Complete
ChromaDB collection	Complete
All 8 documents indexed	Complete
Structured prompt	Complete
Role/context/task/format/length	Complete
Negative constraint	Complete
Few-shot example	Complete
classify_intent node	Complete
retrieve_and_answer node	Complete
direct_answer node	Complete
Conditional routing	Complete
Deterministic mock mode	Complete
Pydantic output	Complete
FastAPI /ask	Complete
Policy API example	Complete
General API example	Complete
Dockerfile	Complete
Docker build/run	Requires Docker installation
Architecture documentation	Complete


## 28. Final Summary
The Support Assistant implements an end-to-end offline RAG pipeline for
Zepto policy questions.
The system ingests eight policy documents, creates local embeddings using
all-MiniLM-L6-v2, and stores them in the zepto_policies ChromaDB
collection.
LangGraph routes policy questions through retrieval and routes unrelated
questions to a direct response node. The final output is validated using a
Pydantic schema containing answer, sources, and confidence.
The FastAPI service exposes the pipeline through POST /ask.
The required baseline works without an LLM API key or paid service by using
the deterministic MOCK_LLM path.

