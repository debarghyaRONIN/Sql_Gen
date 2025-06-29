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

