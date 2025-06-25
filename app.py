from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googletrans import Translator
from typing import Optional, List, Dict, Any
import requests
import re
import logging
import os
import time
from datetime import datetime, timedelta
import threading
import cx_Oracle
import pandas as pd
import json
import uuid
sample = df = pd.read_csv('data.csv')
texts = df.apply(lambda row: ' | '.join(row.values.astype(str)), axis=1).tolist()
from sentence_transformers import SentenceTransformer
import chromadb
client = chromadb.Client()
collection = client.create_collection(name="csv_data")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Or any local model
embeddings = model.encode(texts)
query = "Find records about missing cars"
query_embedding = model.encode([query])

results = collection.query(query_embeddings=query_embedding, n_results=5)
retrieved_texts = [doc for doc in results['documents'][0]]


for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    collection.add(
        ids=[str(i)],
        documents=[text],
        embeddings=[embedding]
    )



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Initialize translator
translator = Translator()

app = FastAPI(
    title="Police Database SQL Generator API - Enhanced",
    description="AI-powered natural language to SQL converter for police CCTNS database with advanced features",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_LANGUAGE = "en"
SESSION_TIMEOUT_MINUTES = 30
ORACLE_CONNECTION_STRING = os.getenv("ORACLE_CONNECTION_STRING", "")

# Enhanced Session Management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.datasets: Dict[str, Dict] = {}  # Store labeled datasets
        self.lock = threading.Lock()
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create new one with provided session ID"""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now(),
                    'chat_history': [],
                    'total_queries': 0,
                    'datasets': [],  # List of dataset IDs created in this session
                    'context': {}  # Store conversation context
                }
            else:
                self.sessions[session_id]['last_accessed'] = datetime.now()
            
            return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by session ID"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if datetime.now() - session['last_accessed'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    del self.sessions[session_id]
                    return None
                
                session['last_accessed'] = datetime.now()
                return session
            return None
    
    def add_to_history(self, session_id: str, query_data: Dict):
        """Add query to session history"""
        session = self.get_or_create_session(session_id)
        with self.lock:
            session['chat_history'].append(query_data)
            session['total_queries'] += 1
    
    def save_dataset(self, dataset_id: str, dataset_info: Dict):
        """Save labeled dataset"""
        with self.lock:
            self.datasets[dataset_id] = dataset_info
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID"""
        return self.datasets.get(dataset_id)
    
    def get_session_datasets(self, session_id: str) -> List[Dict]:
        """Get all datasets for a session"""
        session = self.get_session(session_id)
        if session:
            return [self.datasets[dataset_id] for dataset_id in session['datasets'] 
                   if dataset_id in self.datasets]
        return []

# Database Schema Inspector
class SchemaInspector:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.schema_cache = {}
    
    def connect(self):
        """Create Oracle database connection"""
        try:
            return cx_Oracle.connect(self.connection_string)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    
    def get_tables(self) -> List[Dict]:
        """Get all tables with metadata"""
        if 'tables' in self.schema_cache:
            return self.schema_cache['tables']
        
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT table_name, num_rows, last_analyzed
                    FROM user_tables
                    ORDER BY table_name
                """)
                
                tables = []
                for row in cursor.fetchall():
                    tables.append({
                        'table_name': row[0],
                        'num_rows': row[1] or 0,
                        'last_analyzed': row[2].isoformat() if row[2] else None
                    })
                
                self.schema_cache['tables'] = tables
                return tables
                
        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {str(e)}")
    
    def get_table_columns(self, table_name: str) -> List[Dict]:
        """Get columns for a specific table"""
        cache_key = f"columns_{table_name}"
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT column_name, data_type, nullable, data_default, column_id
                    FROM user_tab_columns
                    WHERE table_name = :table_name
                    ORDER BY column_id
                """, table_name=table_name.upper())
                
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        'column_name': row[0],
                        'data_type': row[1],
                        'nullable': row[2] == 'Y',
                        'default_value': row[3],
                        'column_id': row[4]
                    })
                
                self.schema_cache[cache_key] = columns
                return columns
                
        except Exception as e:
            logger.error(f"Error fetching columns for {table_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch columns: {str(e)}")
    
    def get_foreign_keys(self) -> List[Dict]:
        """Get foreign key relationships"""
        if 'foreign_keys' in self.schema_cache:
            return self.schema_cache['foreign_keys']
        
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        a.table_name as child_table,
                        a.column_name as child_column,
                        c.table_name as parent_table,
                        c.column_name as parent_column,
                        a.constraint_name
                    FROM user_cons_columns a
                    JOIN user_constraints b ON a.constraint_name = b.constraint_name
                    JOIN user_cons_columns c ON b.r_constraint_name = c.constraint_name
                    WHERE b.constraint_type = 'R'
                    ORDER BY a.table_name, a.column_name
                """)
                
                fks = []
                for row in cursor.fetchall():
                    fks.append({
                        'child_table': row[0],
                        'child_column': row[1],
                        'parent_table': row[2],
                        'parent_column': row[3],
                        'constraint_name': row[4]
                    })
                
                self.schema_cache['foreign_keys'] = fks
                return fks
                
        except Exception as e:
            logger.error(f"Error fetching foreign keys: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch foreign keys: {str(e)}")

# Database Execution Agent
class DatabaseExecutor:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def execute_query(self, sql: str, limit: int = 1000) -> Dict:
        """Execute SQL query and return results"""
        try:
            with cx_Oracle.connect(self.connection_string) as conn:
                # Add ROWNUM limit for safety
                if not re.search(r'*<=', sql.upper()) and not re.search(r'LIMIT\s+\d+', sql.upper()):
                    sql = f"SELECT * FROM ({sql}) WHERE <= {limit}"
                
                df = pd.read_sql(sql, conn)
                
                return {
                    'success': True,
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'row_count': len(df),
                    'execution_time': time.time()
                }
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'columns': [],
                'row_count': 0
            }

# Graph Generation Agent
class GraphAgent:
    def __init__(self):
        self.chart_types = ['bar', 'line', 'pie', 'scatter', 'histogram']
    
    def suggest_chart_type(self, data: List[Dict], columns: List[str]) -> str:
        """Suggest appropriate chart type based on data"""
        if not data or not columns:
            return 'table'
        
        # Simple heuristics for chart type suggestion
        numeric_cols = []
        categorical_cols = []
        
        for col in columns:
            sample_values = [row.get(col) for row in data[:10] if row.get(col) is not None]
            if sample_values:
                try:
                    [float(val) for val in sample_values]
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    categorical_cols.append(col)
        
        if len(numeric_cols) >= 2:
            return 'scatter'
        elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            return 'bar'
        elif len(categorical_cols) >= 1:
            return 'pie'
        else:
            return 'table'
    
    def generate_chart_config(self, data: List[Dict], columns: List[str], chart_type: str = None) -> Dict:
        """Generate chart configuration"""
        if not chart_type:
            chart_type = self.suggest_chart_type(data, columns)
        
        config = {
            'type': chart_type,
            'data': data,
            'columns': columns,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False
            }
        }
        
        # Add chart-specific configurations
        if chart_type == 'bar':
            config['options']['scales'] = {
                'y': {'beginAtZero': True}
            }
        elif chart_type == 'pie':
            config['options']['plugins'] = {
                'legend': {'position': 'right'}
            }
        
        return config

# Initialize components
session_manager = SessionManager()
schema_inspector = SchemaInspector(ORACLE_CONNECTION_STRING) if ORACLE_CONNECTION_STRING else None
db_executor = DatabaseExecutor(ORACLE_CONNECTION_STRING) if ORACLE_CONNECTION_STRING else None
graph_agent = GraphAgent()

# Enhanced Pydantic Models
class QueryRequest(BaseModel):
    question: str
    session_id: str
    model: Optional[str] = DEFAULT_MODEL
    source_language: Optional[str] = DEFAULT_LANGUAGE
    target_language: Optional[str] = DEFAULT_LANGUAGE
    execute_query: Optional[bool] = True
    result_limit: Optional[int] = 1000

class DatasetLabelRequest(BaseModel):
    session_id: str
    query_result_id: str
    dataset_name: str
    description: Optional[str] = ""
    tags: Optional[List[str]] = []

class ChartGenerationRequest(BaseModel):
    dataset_id: str
    chart_type: Optional[str] = None
    title: Optional[str] = ""
    x_axis: Optional[str] = ""
    y_axis: Optional[str] = ""

class ConversationalQueryRequest(BaseModel):
    question: str
    session_id: str
    context_dataset_ids: Optional[List[str]] = []
    model: Optional[str] = DEFAULT_MODEL

class SchemaInspectionRequest(BaseModel):
    table_name: Optional[str] = None
    include_relationships: Optional[bool] = True

# Enhanced Response Models
class EnhancedSQLResponse(BaseModel):
    session_id: str
    query_id: str
    question: str
    sql_query: str
    query_tables: List[str]
    query_fields: List[str]
    execution_result: Optional[Dict] = None
    suggested_chart_type: Optional[str] = None
    model_used: str
    generation_time_ms: float
    timestamp: str

class DatasetResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    description: str
    tags: List[str]
    sql_query: str
    data_preview: List[Dict]
    column_info: List[str]
    row_count: int
    created_at: str

class ChartResponse(BaseModel):
    chart_id: str
    dataset_id: str
    chart_config: Dict
    chart_type: str
    title: str
    created_at: str

class SchemaResponse(BaseModel):
    tables: List[Dict]
    foreign_keys: Optional[List[Dict]] = None
    total_tables: int
    schema_version: str

# Enhanced Ollama Client
class EnhancedOllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def generate_sql_with_enhanced_context(self, question: str, chat_history: List[Dict], 
                                         schema_info: Dict, model: str = DEFAULT_MODEL) -> str:
        """Generate SQL with enhanced schema context"""
        
        # Build schema context
        schema_context = self._build_schema_context(schema_info)
        
        # Build conversation context
        conversation_context = ""
        if chat_history:
            conversation_context = "\n\nRecent conversation context:\n"
            for i, entry in enumerate(chat_history[-3:], 1):
                conversation_context += f"{i}. Q: {entry.get('question', '')}\n   SQL: {entry.get('sql_query', '')}\n"

        prompt = f"""
You are an expert Oracle SQL generator specialized in Police CCTNS database systems.

Based on the following data:\n\n{retrieved_texts}\n\nAnswer the query: {query}
{schema_context}
You will receive natural language questions related to law enforcement data and convert them into syntactically correct and optimized Oracle SQL SELECT queries.

Context:
The database schema includes normalized tables such as FIR, OFFICER, POLICE_STATION, DISTRICT_MASTER, VEHICLE_MASTER, ACCUSED, ARREST, and other related entities.

Code-to-name relationships exist and must be resolved using master tables (e.g., DISTRICT_MASTER, OFFICER_MASTER, VEHICLE_TYPE_MASTER, etc.).

STRICT RULES FOR SQL GENERATION
You must follow all of the following rules without exception:

Only generate SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.

Use valid Oracle SQL syntax.

Always use table aliases for readability (e.g., F for FIR, O for OFFICER).

Use JOIN clauses (not implicit joins) with proper ON conditions.

Use WHERE clauses for filtering and ensure filters are relevant to the query.

For date comparisons, always use TO_DATE() with 'YYYY-MM-DD' format. Example:
WHERE F.incident_date BETWEEN TO_DATE('2025-01-01', 'YYYY-MM-DD') AND TO_DATE('2025-01-31', 'YYYY-MM-DD')

Use UPPER() or LOWER() for case-insensitive searches.



Resolve all code values via joins to the appropriate master tables.
Examples:

DISTRICT_CODE → DISTRICT_MASTER

OFFICER_ID → OFFICER_MASTER

Never return explanations or comments. Output only the raw SQL query.

Common SQL Patterns to Follow
District lookup:
JOIN DISTRICT_MASTER D ON PS.district_code = D.district_code

Officer lookup:
JOIN OFFICER_MASTER O ON F.officer_id = O.officer_id

Date range filter:
WHERE F.incident_date BETWEEN TO_DATE('start_date', 'YYYY-MM-DD') AND TO_DATE('end_date', 'YYYY-MM-DD')

Case-insensitive string match:
WHERE UPPER(V.vehicle_type) = 'CAR'

Whenever a natural language question is provided, return only the valid Oracle SQL SELECT query as per the above rules.
{conversation_context}

Human Question: {question}

Oracle SQL Query:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 300
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            sql_query = result.get("response", "").strip()
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            return sql_query
            
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

    def generate_conversational_response(self, question: str, chat_history: List[Dict], 
                                       context_datasets: List[Dict], model: str = DEFAULT_MODEL) -> Dict:
        """Generate conversational response with dataset context"""
        
        # Build dataset context
        dataset_context = ""
        if context_datasets:
            dataset_context = "\n\nAvailable datasets from previous queries:\n"
            for i, dataset in enumerate(context_datasets, 1):
                dataset_context += f"{i}. {dataset['dataset_name']}: {dataset['description']}\n"
                dataset_context += f"   Columns: {', '.join(dataset['column_info'])}\n"
                dataset_context += f"   Rows: {dataset['row_count']}\n"

        # Build conversation context
        conversation_context = ""
        if chat_history:
            conversation_context = "\n\nRecent conversation:\n"
            for entry in chat_history[-3:]:
                conversation_context += f"Q: {entry.get('question', '')}\n"
                if 'response' in entry:
                    conversation_context += f"A: {entry['response']}\n"

        prompt = f"""
You are a helpful assistant for police data analysis. You have access to previous query results and can help users understand their data or suggest follow-up analyses.

{dataset_context}
{conversation_context}

Guidelines:
1. If the question refers to previous results, use the dataset context
2. Suggest relevant follow-up queries when appropriate
3. Explain data patterns or insights when relevant
4. Be conversational and helpful
5. If a new SQL query would be helpful, suggest it

Human Question: {question}

Response:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 400
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "").strip()
            
            return {
                'response': response_text,
                'has_sql_suggestion': 'SELECT' in response_text.upper(),
                'context_used': len(context_datasets) > 0
            }
            
        except requests.RequestException as e:
            logger.error(f"Conversational API error: {e}")
            raise HTTPException(status_code=500, detail=f"Conversational API error: {str(e)}")

    def _build_schema_context(self, schema_info: Dict) -> str:
        """Build schema context for prompts"""
        context = "Database Schema Information:\n\n"
        
        if 'tables' in schema_info:
            context += "Available Tables:\n"
            for table in schema_info['tables']:
                context += f"- {table['table_name']} ({table.get('num_rows', 0)} rows)\n"
        
        if 'foreign_keys' in schema_info and schema_info['foreign_keys']:
            context += "\nKey Relationships:\n"
            for fk in schema_info['foreign_keys'][:10]:  # Limit for prompt size
                context += f"- {fk['child_table']}.{fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}\n"
        
        return context

    def _clean_sql_query(self, sql: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown formatting
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = re.sub(r'^sql\s*', '', sql, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        
        # Validate it's a SELECT statement
        if not sql.upper().startswith('SELECT'):
            raise HTTPException(status_code=400, detail="Query must be a SELECT statement")
        
        # Check for forbidden keywords
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        sql_upper = sql.upper()
        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                raise HTTPException(status_code=400, detail=f"Forbidden SQL keyword: {keyword}")
        
        return sql

    def extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        sql_upper = sql.upper()
        
        # Find FROM clause
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            tables.append(from_match.group(1))
        
        # Find JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend(join_matches)
        
        return list(set(tables))

    def extract_fields_from_sql(self, sql: str) -> List[str]:
        """Extract column fields from SQL query"""
        fields = []
        
        # Extract SELECT fields
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql.upper())
        if select_match:
            select_part = select_match.group(1)
            
            if '*' in select_part:
                return ['*']
            
            # Split by comma and clean field names
            field_parts = select_part.split(',')
            for field in field_parts:
                field = field.strip()
                if '.' in field:
                    field = field.split('.')[-1]
                if ' AS ' in field.upper():
                    field = field.split(' AS ')[0].strip()
                if field and field != '*':
                    fields.append(field.lower())
        
        return list(set(fields))

# Initialize enhanced client
enhanced_ollama_client = EnhancedOllamaClient()

# Background session cleanup (same as before)
def cleanup_sessions_periodically():
    """Periodic cleanup of expired sessions"""
    def cleanup():
        while True:
            try:
                cleaned = session_manager.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired sessions")
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                time.sleep(60)
    
    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()

cleanup_sessions_periodically()

# Enhanced API Routes

@app.post("/schema/inspect", response_model=SchemaResponse)
async def inspect_schema(request: SchemaInspectionRequest):
    """Inspect database schema with automatic ingestion"""
    if not schema_inspector:
        raise HTTPException(status_code=500, detail="Database connection not configured")
    
    try:
        tables = schema_inspector.get_tables()
        foreign_keys = None
        
        if request.include_relationships:
            foreign_keys = schema_inspector.get_foreign_keys()
        
        # If specific table requested, get column details
        if request.table_name:
            table_columns = schema_inspector.get_table_columns(request.table_name)
            return SchemaResponse(
                tables=[{
                    'table_name': request.table_name,
                    'columns': table_columns
                }],
                foreign_keys=foreign_keys,
                total_tables=1,
                schema_version=datetime.now().isoformat()
            )
        
        return SchemaResponse(
            tables=tables,
            foreign_keys=foreign_keys,
            total_tables=len(tables),
            schema_version=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Schema inspection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/enhanced-sql", response_model=EnhancedSQLResponse)
async def generate_enhanced_sql(request: QueryRequest):
    """Enhanced SQL generation with schema understanding and execution"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Get session and schema info
        session = session_manager.get_or_create_session(request.session_id)
        
        schema_info = {}
        if schema_inspector:
            schema_info = {
                'tables': schema_inspector.get_tables(),
                'foreign_keys': schema_inspector.get_foreign_keys()
            }
        
        # Generate SQL with enhanced context
        sql_query = enhanced_ollama_client.generate_sql_with_enhanced_context(
            request.question,
            session['chat_history'],
            schema_info,
            request.model
        )
        
        # Extract metadata
        query_tables = enhanced_ollama_client.extract_tables_from_sql(sql_query)
        query_fields = enhanced_ollama_client.extract_fields_from_sql(sql_query)
        
        # Execute query if requested and possible
        execution_result = None
        suggested_chart_type = None
        
        if request.execute_query and db_executor:
            execution_result = db_executor.execute_query(sql_query, request.result_limit)
            
            if execution_result['success'] and execution_result['data']:
                suggested_chart_type = graph_agent.suggest_chart_type(
                    execution_result['data'], 
                    execution_result['columns']
                )
        
        generation_time = (time.time() - start_time) * 1000
        current_timestamp = datetime.now().isoformat()
        
        response_data = EnhancedSQLResponse(
            session_id=request.session_id,
            query_id=query_id,
            question=request.question,
            sql_query=sql_query,
            query_tables=query_tables,
            query_fields=query_fields,
            execution_result=execution_result,
            suggested_chart_type=suggested_chart_type,
            model_used=request.model,
            generation_time_ms=round(generation_time, 2),
            timestamp=current_timestamp
        )
        
        # Add to session history
        history_entry = {
            "query_id": query_id,
            "question": request.question,
            "sql_query": sql_query,
            "query_tables": query_tables,
            "query_fields": query_fields,
            "execution_success": execution_result['success'] if execution_result else None,
            "row_count": execution_result['row_count'] if execution_result else 0,
            "timestamp": current_timestamp
        }
        session_manager.add_to_history(request.session_id, history_entry)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Enhanced SQL generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/label", response_model=DatasetResponse)
async def label_dataset(request: DatasetLabelRequest):
    """Label and save query results as a dataset"""
    try:
        # Get the query result from session history
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Find the query result
        query_result = None
        for entry in session['chat_history']:
            if entry.get('query_id') == request.query_result_id:
                query_result = entry
                break
        
        if not query_result:
            raise HTTPException(status_code=404, detail="Query result not found")
        
        # Create dataset
        dataset_id = str(uuid.uuid4())
        dataset_info = {
            'dataset_id': dataset_id,
            'dataset_name': request.dataset_name,
            'description': request.description,
            'tags': request.tags,
            'sql_query': query_result['sql_query'],
            'session_id': request.session_id,
            'created_at': datetime.now().isoformat(),
            'query_tables': query_result.get('query_tables', []),
            'query_fields': query_result.get('query_fields', [])
        }
        
        # Execute query to get fresh data
        if db_executor:
            execution_result = db_executor.execute_query(query_result['sql_query'])
            dataset_info.update({
                'data': execution_result['data'],
                'columns': execution_result['columns'],
                'row_count': execution_result['row_count']
            })
        
        # Save dataset
        session_manager.save_dataset(dataset_id, dataset_info)
        
        # Add to session datasets
        session['datasets'].append(dataset_id)
        
        return DatasetResponse(
            dataset_id=dataset_id,
            dataset_name=request.dataset_name,
            description=request.description,
            tags=request.tags,
            sql_query=query_result['sql_query'],
            data_preview=dataset_info.get('data', [])[:10],  # First 10 rows
            column_info=dataset_info.get('columns', []),
            row_count=dataset_info.get('row_count', 0),
            created_at=dataset_info['created_at']
        )
        
    except Exception as e:
        logger.error(f"Dataset labeling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/charts/generate", response_model=ChartResponse)
async def generate_chart(request: ChartGenerationRequest):
    """Generate chart configuration for a dataset"""
    try:
        # Get dataset
        dataset = session_manager.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Generate chart configuration
        chart_config = graph_agent.generate_chart_config(
            dataset.get('data', []),
            dataset.get('columns', []),
            request.chart_type
        )
        
        # Add custom configurations
        if request.title:
            chart_config['options']['plugins'] = chart_config['options'].get('plugins', {})
            chart_config['options']['plugins']['title'] = {
                'display': True,
                'text': request.title
            }
        
        if request.x_axis or request.y_axis:
            chart_config['options']['scales'] = chart_config['options'].get('scales', {})
            if request.x_axis:
                chart_config['options']['scales']['x'] = {
                    'title': {'display': True, 'text': request.x_axis}
                }
            if request.y_axis:
                chart_config['options']['scales']['y'] = {
                    'title': {'display': True, 'text': request.y_axis}
                }
        
        chart_id = str(uuid.uuid4())
        
        return ChartResponse(
            chart_id=chart_id,
            dataset_id=request.dataset_id,
            chart_config=chart_config,
            chart_type=chart_config['type'],
            title=request.title or f"Chart for {dataset.get('dataset_name', 'Dataset')}",
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/conversational")
async def conversational_query(request: ConversationalQueryRequest):
    """Handle conversational queries with context awareness"""
    try:
        # Get session and context datasets
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context_datasets = []
        if request.context_dataset_ids:
            for dataset_id in request.context_dataset_ids:
                dataset = session_manager.get_dataset(dataset_id)
                if dataset:
                    context_datasets.append(dataset)
        
        # Generate conversational response
        response_data = enhanced_ollama_client.generate_conversational_response(
            request.question,
            session['chat_history'],
            context_datasets,
            request.model
        )
        
        # Add to session history
        history_entry = {
            "type": "conversational",
            "question": request.question,
            "response": response_data['response'],
            "context_datasets": request.context_dataset_ids,
            "timestamp": datetime.now().isoformat()
        }
        session_manager.add_to_history(request.session_id, history_entry)
        
        return {
            "response": response_data['response'],
            "has_sql_suggestion": response_data['has_sql_suggestion'],
            "context_used": response_data['context_used'],
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Conversational query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session chat history"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "created_at": session['created_at'].isoformat(),
        "last_accessed": session['last_accessed'].isoformat(),
        "total_queries": session['total_queries'],
        "chat_history": session['chat_history'][-20:],  # Last 20 entries
        "datasets": session.get('datasets', [])
    }

@app.get("/sessions/{session_id}/datasets")
async def get_session_datasets(session_id: str):
    """Get all datasets for a session"""
    datasets = session_manager.get_session_datasets(session_id)
    return {
        "session_id": session_id,
        "datasets": datasets,
        "total_datasets": len(datasets)
    }

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details"""
    dataset = session_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    try:
        if dataset_id not in session_manager.datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Remove from datasets
        del session_manager.datasets[dataset_id]
        
        # Remove from all sessions
        for session in session_manager.sessions.values():
            if dataset_id in session.get('datasets', []):
                session['datasets'].remove(dataset_id)
        
        return {"message": "Dataset deleted successfully", "dataset_id": dataset_id}
        
    except Exception as e:
        logger.error(f"Dataset deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        ollama_status = "connected"
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                ollama_status = "disconnected"
        except:
            ollama_status = "disconnected"
        
        # Test database connection
        db_status = "not_configured"
        if ORACLE_CONNECTION_STRING:
            try:
                if schema_inspector:
                    with schema_inspector.connect() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1 FROM DUAL")
                        cursor.fetchone()
                    db_status = "connected"
            except:
                db_status = "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "ollama": ollama_status,
                "database": db_status
            },
            "active_sessions": len(session_manager.sessions),
            "total_datasets": len(session_manager.datasets)
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "active_sessions": len(session_manager.sessions),
        "total_datasets": len(session_manager.datasets),
        "database_configured": ORACLE_CONNECTION_STRING is not None,
        "ollama_url": OLLAMA_BASE_URL,
        "default_model": DEFAULT_MODEL,
        "session_timeout_minutes": SESSION_TIMEOUT_MINUTES,
        "uptime": time.time()
    }

# Add missing cleanup method to SessionManager
def cleanup_expired_sessions_method(self):
    """Clean up expired sessions"""
    with self.lock:
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_accessed'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)

# Add the method to SessionManager class
SessionManager.cleanup_expired_sessions = cleanup_expired_sessions_method

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Police Database SQL Generator API - Enhanced",
        "version": "2.0.0",
        "description": "AI-powered natural language to SQL converter for police CCTNS database with advanced features",
        "features": [
            "Natural language to SQL conversion",
            "Database schema inspection",
            "Query execution and results",
            "Dataset labeling and management",
            "Chart generation",
            "Conversational interface",
            "Session management",
            "Multi-language support"
        ],
        "endpoints": {
            "schema": "/schema/inspect",
            "sql_generation": "/query/enhanced-sql",
            "dataset_labeling": "/datasets/label",
            "chart_generation": "/charts/generate",
            "conversational": "/chat/conversational",
            "health": "/health",
            "stats": "/stats"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )