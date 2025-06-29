# Police Database SQL Generator API

## Overview

This project is an advanced, AI-powered API platform designed to convert natural language queries into SQL for police CCTNS (Crime and Criminal Tracking Network & Systems) databases. It provides a rich set of features for law enforcement data analysis, including schema inspection, query execution, dataset labeling, chart generation, conversational querying, and robust session management. The backend leverages FastAPI, advanced LLMs (such as Gemini and Ollama), and supports Oracle SQL as well as CSV/ChromaDB for semantic search.

---

## Features

- **Natural Language to SQL Conversion**: Converts user questions into syntactically correct and optimized SQL queries for police databases.
- **Database Schema Inspection**: Explore tables, columns, and relationships in the police database.
- **Query Execution & Results**: Run generated SQL queries and retrieve results directly from the database.
- **Dataset Labeling & Management**: Save, label, and manage query results as reusable datasets.
- **Chart Generation**: Automatically suggest and generate chart configurations for data visualization.
- **Conversational Interface**: Context-aware chat for follow-up questions and data exploration.
- **Session Management**: Track user sessions, chat history, and datasets.
- **Multi-language Support**: Translate queries and results for multilingual users.
- **Semantic Search**: Use ChromaDB and Sentence Transformers for semantic retrieval from CSV data.

---

## Architecture

- **FastAPI**: Main web framework for API endpoints.
- **Ollama/Gemini LLMs**: Used for natural language understanding and SQL generation.
- **Oracle Database**: Main backend for police data (can be replaced with other RDBMS).
- **ChromaDB + Sentence Transformers**: For semantic search over CSV data.
- **Pandas**: For data manipulation and preview.
- **Pydantic**: For request/response validation.
- **Threading**: For background session cleanup.

---

## API Endpoints

### Root
- `GET /` — API information and available endpoints.

### Schema Inspection
- `POST /schema/inspect` — Inspect database schema, tables, columns, and relationships.

### SQL Generation
- `POST /query/enhanced-sql` — Generate SQL from natural language, execute, and return results.

### Dataset Management
- `POST /datasets/label` — Label and save query results as datasets.
- `GET /datasets/{dataset_id}` — Get dataset details.
- `DELETE /datasets/{dataset_id}` — Delete a dataset.
- `GET /sessions/{session_id}/datasets` — List all datasets for a session.

### Chart Generation
- `POST /charts/generate` — Generate chart configuration for a dataset.

### Conversational Query
- `POST /chat/conversational` — Conversational interface for follow-up questions and context-aware queries.

### Session Management
- `GET /sessions/{session_id}/history` — Get chat history for a session.

### Health & Stats
- `GET /health` — Health check for API and services.
- `GET /stats` — System statistics.

---

## Example Usage

### 1. Generate SQL from Natural Language
```json
POST /query/enhanced-sql
{
  "question": "Show all FIRs registered in 2025 in Hyderabad district",
  "session_id": "abc123"
}
```

### 2. Inspect Database Schema
```json
POST /schema/inspect
{
  "table_name": "FIR",
  "include_relationships": true
}
```

### 3. Label a Dataset
```json
POST /datasets/label
{
  "session_id": "abc123",
  "query_result_id": "result-uuid",
  "dataset_name": "Hyderabad FIRs 2025",
  "description": "All FIRs in Hyderabad for 2025"
}
```

### 4. Generate a Chart
```json
POST /charts/generate
{
  "dataset_id": "dataset-uuid",
  "chart_type": "bar",
  "title": "FIRs by Month"
}
```

### 5. Conversational Query
```json
POST /chat/conversational
{
  "question": "How many of these FIRs are for theft?",
  "session_id": "abc123",
  "context_dataset_ids": ["dataset-uuid"]
}
```

---

## Session & Dataset Management
- Sessions are tracked by `session_id` and store chat history, queries, and datasets.
- Datasets are labeled query results, reusable for further analysis and charting.
- Background cleanup removes expired sessions.

---

## Semantic Search (CSV/ChromaDB)
- CSV data is loaded and embedded using Sentence Transformers.
- ChromaDB enables semantic retrieval for queries like "Find records about missing cars".

---

## Configuration
- **Environment Variables:**
  - `OLLAMA_BASE_URL`: URL for Ollama LLM API (default: `http://localhost:11434`)
  - `OLLAMA_MODEL`: Default LLM model (e.g., `llama3.2:3b`)
  - `ORACLE_CONNECTION_STRING`: Oracle DB connection string

---

## Running the API

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables** (if needed):
   ```bash
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_MODEL=llama3.2:3b
   export ORACLE_CONNECTION_STRING=your_oracle_conn_str
   ```
3. **Start the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Project Structure

```
API/
  sql-api/
    app.py           # Main FastAPI application
    requirements.txt # Python dependencies
    README.md        # This documentation
    ...
```

---

## Extending the Project
- Add new endpoints for analytics, reporting, or admin features.
- Integrate with other LLMs or RDBMS.
- Enhance semantic search with more data sources.

---
graph TD
    A[Client Request] --> B{Request Type?}
    
    B -->|Schema Inspection| C[/schema/inspect]
    B -->|SQL Generation| D[/query/enhanced-sql]
    B -->|Dataset Operations| E[/datasets/*]
    B -->|Chart Generation| F[/charts/generate]
    B -->|Conversational| G[/chat/conversational]
    B -->|Health Check| H[/health]
    
    %% Schema Inspection Flow
    C --> I[SchemaInspector]
    I --> I1[Connect to Oracle DB]
    I1 --> I2[Get Tables & Metadata]
    I2 --> I3[Get Foreign Keys]
    I3 --> I4[Cache Schema Info]
    I4 --> I5[Return Schema Response]
    
    %% SQL Generation Flow
    D --> J[SessionManager]
    J --> J1[Get/Create Session]
    J1 --> J2[Build Chat History Context]
    J2 --> K[EnhancedOllamaClient]
    K --> K1[Load CSV Data]
    K1 --> K2[Generate Embeddings with SentenceTransformer]
    K2 --> K3[Store in ChromaDB]
    K3 --> K4[Query ChromaDB for Context]
    K4 --> K5[Build Enhanced Prompt with Schema + Context]
    K5 --> K6[Call Ollama API]
    K6 --> K7[Clean & Validate SQL]
    K7 --> K8[Extract Tables & Fields]
    K8 --> L{Execute Query?}
    
    L -->|Yes| M[DatabaseExecutor]
    L -->|No| N[Return SQL Only]
    
    M --> M1[Execute Oracle SQL]
    M1 --> M2[Add ROWNUM Limit for Safety]
    M2 --> M3[Return Results as DataFrame]
    M3 --> O[GraphAgent]
    O --> O1[Suggest Chart Type]
    O1 --> P[Update Session History]
    P --> Q[Return Enhanced SQL Response]
    
    %% Dataset Operations Flow
    E --> R{Dataset Operation?}
    R -->|Label Dataset| S[/datasets/label]
    R -->|Get Dataset| T[/datasets/{id}]
    R -->|Delete Dataset| U[/datasets/{id} DELETE]
    
    S --> S1[Find Query Result in Session]
    S1 --> S2[Execute Fresh Query]
    S2 --> S3[Create Dataset with UUID]
    S3 --> S4[Save to SessionManager]
    S4 --> S5[Return Dataset Response]
    
    %% Chart Generation Flow
    F --> V[Get Dataset from SessionManager]
    V --> W[GraphAgent.generate_chart_config]
    W --> W1[Analyze Data Types]
    W1 --> W2[Suggest Chart Type]
    W2 --> W3[Build Chart.js Config]
    W3 --> W4[Return Chart Response]
    
    %% Conversational Flow
    G --> X[Get Session & Context Datasets]
    X --> Y[EnhancedOllamaClient.generate_conversational_response]
    Y --> Y1[Build Dataset Context]
    Y1 --> Y2[Build Conversation History]
    Y2 --> Y3[Generate Contextual Response]
    Y3 --> Y4[Update Session History]
    Y4 --> Y5[Return Conversational Response]
    
    %% Core Components
    subgraph "Core Components"
        AA[SessionManager]
        BB[SchemaInspector]
        CC[DatabaseExecutor]
        DD[GraphAgent]
        EE[EnhancedOllamaClient]
    end
    
    %% Data Storage
    subgraph "Data Storage"
        FF[(Oracle Database)]
        GG[ChromaDB Vector Store]
        HH[In-Memory Sessions]
        II[CSV Data Files]
    end
    
    %% External Services
    subgraph "External Services"
        JJ[Ollama API Server]
        KK[SentenceTransformer Model]
        LL[Google Translate API]
    end
    
    %% Data Flow Connections
    I1 -.-> FF
    M1 -.-> FF
    K2 -.-> KK
    K3 -.-> GG
    K4 -.-> GG
    K6 -.-> JJ
    J1 -.-> HH
    S4 -.-> HH
    
    %% Background Processes
    subgraph "Background Processes"
        MM[Session Cleanup Thread]
        NN[Periodic Cache Updates]
    end
    
    MM -.-> HH
    NN -.-> I4
    
    %% Key Features Highlight
    subgraph "Key Features"
        OO[Multi-language Support]
        PP[RAG with Vector Search]
        QQ[Schema-aware SQL Generation]
        RR[Query Result Caching]
        SS[Conversational Interface]
        TT[Chart Generation]
    end
    
    %% Security & Safety
    subgraph "Security Features"
        UU[SQL Injection Prevention]
        VV[Query Validation]
        WW[SELECT-only Restriction]
        XX[Session Timeout]
    end
    
    K7 -.-> UU
    K7 -.-> VV
    K7 -.-> WW
    MM -.-> XX
    
    %% Styling
    classDef apiEndpoint fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef coreComponent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef dataStore fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef externalService fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef securityFeature fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    
    class C,D,E,F,G,H apiEndpoint
    class AA,BB,CC,DD,EE coreComponent
    class FF,GG,HH,II dataStore
    class JJ,KK,LL externalService
    class UU,VV,WW,XX securityFeature
## License
This project is for demonstration and research purposes. Please check with your organization for deployment in production environments.

---

