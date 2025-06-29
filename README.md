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

## License
This project is for demonstration and research purposes. Please check with your organization for deployment in production environments.

---

```mermaid
graph TD
    A[Client Request] --> B{Request Type}
    
    %% Main API Endpoints
    B --> C[Schema Inspection]
    B --> D[SQL Generation]
    B --> E[Dataset Operations]
    B --> F[Chart Generation]
    B --> G[Chat Interface]
    
    %% Schema Flow
    C --> C1[Connect to Oracle DB]
    C1 --> C2[Get Tables & Metadata]
    C2 --> C3[Return Schema Info]
    
    %% SQL Generation Flow
    D --> D1[Session Manager]
    D1 --> D2[Enhanced Ollama Client]
    D2 --> D3[Generate SQL with RAG]
    D3 --> D4{Execute Query?}
    D4 -->|Yes| D5[Database Executor]
    D4 -->|No| D6[Return SQL Only]
    D5 --> D7[Return Results]
    
    %% Dataset Operations
    E --> E1[Create/Get/Delete Dataset]
    E1 --> E2[Save to Session]
    
    %% Chart Generation
    F --> F1[Get Dataset]
    F1 --> F2[Analyze Data Types]
    F2 --> F3[Generate Chart Config]
    
    %% Chat Interface
    G --> G1[Build Context]
    G1 --> G2[Generate Response]
    
    %% Core Components
    subgraph Core["Core Components"]
        H1[Session Manager]
        H2[Schema Inspector]
        H3[Database Executor]
        H4[Graph Agent]
        H5[Enhanced Ollama Client]
    end
    
    %% Data Storage
    subgraph Storage["Data Storage"]
        I1[(Oracle Database)]
        I2[Vector Store ChromaDB]
        I3[In-Memory Sessions]
    end
    
    %% External Services
    subgraph External["External Services"]
        J1[Ollama API]
        J2[SentenceTransformer]
    end
    
    %% Key Features
    subgraph Features["Key Features"]
        K1[RAG Vector Search]
        K2[Multi-language Support]
        K3[SQL Validation]
        K4[Session Management]
    end
    
    %% Connect flows to components
    D2 -.-> H5
    D5 -.-> H3
    C1 -.-> H2
    F2 -.-> H4
    D1 -.-> H1
    
    %% Connect to storage
    C1 -.-> I1
    D5 -.-> I1
    D3 -.-> I2
    D1 -.-> I3
    
    %% Connect to external services
    D3 -.-> J1
    D3 -.-> J2
    
    %% Styling (Black blocks, White text)
    classDef endpoint fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px
    classDef process fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px
    classDef storage fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px
    classDef external fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px
    classDef feature fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px
    classDef core fill:#000000,color:#ffffff,stroke:#ffffff,stroke-width:1.5px

    class C,D,E,F,G endpoint
    class C1,C2,C3,D1,D2,D3,D4,D5,D6,D7,E1,E2,F1,F2,F3,G1,G2 process
    class H1,H2,H3,H4,H5 core
    class I1,I2,I3 storage
    class J1,J2 external
    class K1,K2,K3,K4 feature
