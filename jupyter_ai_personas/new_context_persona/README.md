# New Context Retrieval Persona

A sophisticated PocketFlow-based context retrieval persona that provides advanced RAG capabilities for analyzing Jupyter notebooks and retrieving relevant documentation from the Python Data Science Handbook.

## 🏗️ Architecture

This persona uses **PocketFlow architecture** instead of multi-agent systems, providing a more modular and efficient approach to context retrieval.

### Core Components

#### 1. PocketFlow Base Classes (`pocketflow.py`)
- **Flow**: Orchestrates node execution and routing
- **Node**: Base class for all processing nodes
- **ConditionalNode**: Supports conditional routing
- **BatchNode**: Processes data in batches
- **UtilityFunctions**: Helper functions for common operations

#### 2. RAG Nodes (`rag_nodes.py`)
- **SetupRepositoryNode**: Clones/updates Python Data Science Handbook
- **ExtractDocumentsNode**: Extracts content from Jupyter notebooks
- **ChunkDocumentsNode**: Splits documents into manageable chunks
- **EmbedDocumentsNode**: Creates vector embeddings
- **CreateVectorStoreNode**: Builds and persists vector database
- **QueryEmbeddingNode**: Embeds user queries
- **RetrieveDocumentsNode**: Retrieves relevant documents
- **GenerateResponseNode**: Generates final responses

#### 3. Notebook Analysis (`notebook_analyzer.py`)
- **NotebookAnalysisNode**: Analyzes notebook content and context
- **ContextSearchNode**: Creates context-aware search queries
- **NotebookReaderTool**: Compatibility layer for existing interfaces

#### 4. Flow Orchestration (`rag_flows.py`)
- **IndexingFlow**: Offline flow for building vector store
- **RetrievalFlow**: Online flow for query processing
- **ContextRetrievalFlow**: Complete flow with notebook analysis
- **ReportGenerationNode**: Creates comprehensive markdown reports

#### 5. Main Persona (`context_persona.py`)
- **ContextRetrievalAgent**: PocketFlow-based agent
- **NewContextPersona**: Jupyter AI persona integration

## 🚀 Features

### Advanced Context Analysis
- **Notebook Analysis**: Extracts libraries, analysis stage, objectives
- **Query Intent Classification**: Determines user intent (learning, troubleshooting, etc.)
- **Context-Aware Search**: Generates targeted search queries based on context

### RAG Capabilities
- **Semantic Search**: Vector-based search through Python Data Science Handbook
- **Batch Processing**: Efficient processing of large document collections
- **Persistent Storage**: Reusable vector database with Chroma

### Intelligent Reporting
- **Comprehensive Reports**: Detailed markdown reports with actionable insights
- **Code Examples**: Relevant code snippets based on analysis stage
- **Next Steps**: Prioritized recommendations for immediate action

## 🛠️ Installation

### Dependencies
```bash
pip install langchain sentence-transformers chromadb nbformat
```

### Optional Dependencies
```bash
pip install huggingface-hub transformers torch
```

## 📊 Usage

### Basic Usage
```python
from jupyter_ai_personas.new_context_persona import NewContextPersona

# In Jupyter AI chat:
@NewContextPersona analyze my data visualization approach

# With specific notebook:
@NewContextPersona notebook: /path/to/analysis.ipynb help me optimize my pandas operations
```

### Programmatic Usage
```python
from jupyter_ai_personas.new_context_persona import ContextRetrievalAgent

# Initialize agent
agent = ContextRetrievalAgent()

# Ensure vector store is available
agent.ensure_vector_store()

# Run context retrieval
result = agent.run_context_retrieval(
    user_query="How to improve pandas performance",
    notebook_path="/path/to/notebook.ipynb"
)
```

## 🔧 Configuration

### Vector Store Setup
The persona automatically manages the vector store:
- **Location**: `new_context_persona/vector_stores/python_ds_handbook/`
- **Auto-creation**: Creates vector store on first use
- **Persistence**: Reuses existing vector store for faster responses

### Notebook Analysis
- **Auto-detection**: Finds notebook paths in user messages
- **Fallback**: Uses default notebook if none specified
- **Context Extraction**: Analyzes libraries, stages, and objectives

## 🔄 Workflows

### 1. Offline Indexing (IndexingFlow)
```
SetupRepository → ExtractDocuments → ChunkDocuments → EmbedDocuments → CreateVectorStore
```

### 2. Online Retrieval (RetrievalFlow)
```
QueryEmbedding → RetrieveDocuments → GenerateResponse
```

### 3. Context Retrieval (ContextRetrievalFlow)
```
NotebookAnalysis → ContextSearch → ReportGeneration
```

## 📈 Performance

### Efficiency Features
- **Batch Processing**: Handles large document collections efficiently
- **Persistent Storage**: Avoids re-indexing on subsequent runs
- **Caching**: Reuses embeddings and vector stores
- **Lazy Loading**: Only loads components when needed

### Scalability
- **Modular Design**: Easy to add new nodes and flows
- **Configurable Parameters**: Adjustable chunk sizes, embedding models
- **Error Handling**: Graceful fallbacks for missing dependencies

## 🧪 Testing

### Basic Test
```python
from jupyter_ai_personas.new_context_persona import ContextRetrievalAgent

agent = ContextRetrievalAgent()
status = agent.get_status()
print(f"Agent status: {status}")
```

### Flow Test
```python
from jupyter_ai_personas.new_context_persona import ContextRetrievalFlow

flow = ContextRetrievalFlow()
result = flow.run_context_retrieval(
    user_query="pandas dataframe operations",
    notebook_path=None
)
```

## 🔍 Troubleshooting

### Common Issues

#### "Vector store not available"
- **Cause**: First run or missing dependencies
- **Solution**: Install dependencies and allow initial indexing

#### "Notebook not found"
- **Cause**: Invalid notebook path
- **Solution**: Check path or let system use default

#### "Embedding failed"
- **Cause**: Missing sentence-transformers
- **Solution**: `pip install sentence-transformers`

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🆚 Comparison with Original

### Original Context Persona
- **Architecture**: Multi-agent system (3 agents)
- **Framework**: Agno agent framework
- **Complexity**: Higher coordination overhead
- **Dependencies**: Agno, AWS Bedrock

### New Context Persona
- **Architecture**: PocketFlow-based flows
- **Framework**: PocketFlow nodes and flows
- **Complexity**: Streamlined processing pipeline
- **Dependencies**: LangChain, local embeddings

### Benefits of New Architecture
1. **Modularity**: Easy to add/modify processing steps
2. **Efficiency**: Streamlined processing without agent coordination
3. **Flexibility**: Supports different flow configurations
4. **Maintainability**: Clear separation of concerns
5. **Scalability**: Better handling of large document collections

## 🔮 Future Enhancements

### Planned Features
- **Multiple Data Sources**: Support for additional documentation sources
- **Custom Embeddings**: Support for domain-specific embedding models
- **Advanced Analytics**: More sophisticated notebook analysis
- **Integration**: Better integration with other personas

### Extensibility
- **Custom Nodes**: Easy to add new processing nodes
- **Flow Variants**: Support for different analysis workflows
- **Tool Integration**: Integration with external tools and APIs

## 📄 File Structure

```
new_context_persona/
├── __init__.py              # Package initialization
├── README.md                # This documentation
├── pocketflow.py            # Core PocketFlow classes
├── rag_nodes.py             # RAG processing nodes
├── rag_flows.py             # Flow orchestration
├── notebook_analyzer.py     # Notebook analysis components
├── context_persona.py       # Main persona implementation
└── vector_stores/           # Vector database storage
    └── python_ds_handbook/   # Handbook vector store
```

## 🤝 Contributing

To extend this persona:

1. **Add New Nodes**: Create new processing nodes in `rag_nodes.py`
2. **Modify Flows**: Update flow configurations in `rag_flows.py`
3. **Enhance Analysis**: Improve notebook analysis in `notebook_analyzer.py`
4. **Test Changes**: Ensure all flows work correctly

## 📊 Metrics

The persona tracks various metrics:
- **Indexing Performance**: Documents processed, time taken
- **Retrieval Accuracy**: Relevant documents found
- **Analysis Coverage**: Notebook features analyzed
- **Response Quality**: Comprehensive reports generated

---

**🎯 Ready to analyze your data science projects with advanced PocketFlow-based context retrieval!**