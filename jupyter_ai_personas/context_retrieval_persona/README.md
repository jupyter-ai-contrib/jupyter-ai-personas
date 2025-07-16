# Context Retrieval Persona

A sophisticated Jupyter AI persona that analyzes your data science notebooks and provides contextual recommendations using Retrieval-Augmented Generation (RAG) from the Python Data Science Handbook.

## Overview

The Context Retriever Persona is a multi-agent system that understands your current data science work and finds relevant resources from the comprehensive Python Data Science Handbook using semantic search. It consists of three specialized agents working together to provide actionable insights.

## Features

- **Notebook Analysis**: Automatically extracts context from your Jupyter notebooks including libraries, analysis stage, and objectives
- **RAG-Powered Search**: Semantic search through the entire Python Data Science Handbook repository
- **Context-Aware Recommendations**: Provides relevant code examples, best practices, and documentation based on your current work
- **Multi-Agent Architecture**: Three specialized agents for analysis, search, and report generation
- **Comprehensive Reports**: Generates detailed markdown reports with actionable next steps
- **Enhanced Chunk Display**: Full retrieved text chunks are displayed in terminal for debugging
- **Automatic Report Saving**: Generated reports are automatically saved as `repo_context.md`
- **Improved RAG Parameters**: Increased chunk size (1500 chars) and search results (8 chunks) for better coverage

## Architecture

### Three-Agent System

1. **NotebookAnalyzer**: Extracts context from your notebook content
   - Identifies libraries being used (pandas, numpy, scikit-learn, etc.)
   - Determines analysis stage (data loading, EDA, preprocessing, modeling, etc.)
   - Extracts objectives and current progress

2. **KnowledgeSearcher**: Performs targeted RAG searches
   - Multiple search strategies based on context
   - Semantic search through 100+ handbook notebooks
   - Filters for relevant code examples and explanations

3. **MarkdownGenerator**: Creates comprehensive reports
   - Executive summaries of findings
   - Relevant code examples with explanations
   - Actionable next steps for your analysis

## Core Components

### Context Retriever Persona (`context_retriever_persona.py`)
Main persona class that orchestrates the three-agent system and handles Jupyter AI integration.

### RAG Core System (`rag_core.py`)
- Repository management for Python Data Science Handbook
- Document extraction from Jupyter notebooks
- Vector storage using ChromaDB
- Semantic search with HuggingFace embeddings

### RAG Integration Tool (`rag_integration_tool.py`)
Agno tool wrapper providing clean integration with the agent system:
- `search_repository()`: General semantic search
- `search_by_topic()`: Topic-specific searches
- `search_code_examples()`: Code-focused searches

### Notebook Reader Tool (`file_reader_tool.py`)
Comprehensive notebook content extraction:
- Reads all cell types (code, markdown)
- Extracts outputs and metadata
- Detects libraries and analysis patterns
- Provides structured context for search

## Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install chromadb sentence-transformers langchain nbformat gitpython
```

### Quick Setup
```bash
# Run the setup script
python setup_rag_system.py
```

This will:
1. Check dependencies
2. Clone the Python Data Science Handbook repository
3. Build the vector store (first run takes 5-10 minutes)
4. Test the system functionality

### Manual Setup
```python
from rag_core import create_handbook_rag

# Initialize the RAG system
rag = create_handbook_rag(force_rebuild=False)

# Test search functionality
results = rag.search("pandas dataframe operations", k=5)
```

## Usage

### Basic Usage
In Jupyter AI, activate the Context Retriever Persona and provide:

```
I need help with data visualization using matplotlib and seaborn. 
notebook: /path/to/my/analysis.ipynb
```

### Typical Workflow
1. **Context Analysis**: The system reads your notebook to understand:
   - What libraries you're using
   - What stage of analysis you're in
   - What data you're working with

2. **Knowledge Search**: Performs multiple targeted searches:
   - Library-specific examples
   - Analysis stage best practices
   - Problem domain patterns

3. **Report Generation**: Creates a comprehensive markdown report with:
   - Executive summary of findings
   - Current notebook analysis
   - Relevant code examples
   - Actionable next steps

### Example Output
```markdown
## Executive Summary
Based on your notebook analysis, you're in the exploratory data analysis stage 
using pandas and matplotlib. Found relevant handbook content for data 
visualization best practices and statistical analysis patterns.

## Current Notebook Analysis
- Libraries: pandas, matplotlib, seaborn
- Analysis Stage: exploratory_data_analysis
- Data Operations: groupby, pivot, plotting

## Relevant Resources
### Data Visualization with Matplotlib
[Code examples and explanations from the handbook]

### Statistical Analysis Patterns
[Relevant statistical methods and implementations]

## Actionable Next Steps
1. Implement correlation analysis using the patterns from Section 04.05
2. Consider using seaborn for advanced statistical plots
3. Apply dimensionality reduction techniques from Chapter 05
```

## Configuration

### Environment Variables
```bash
# Optional: Configure data paths
export RAG_REPO_PATH="/path/to/PythonDataScienceHandbook"
export RAG_VECTOR_STORE_PATH="/path/to/vector_stores"
```

### Customization
Modify parameters in `rag_core.py`:
```python
rag = PythonDSHandbookRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1500,         # Increased chunk size
    chunk_overlap=300        # Increased overlap
)
```

### RAG Search Parameters
- **Default Results**: 8 chunks per search (increased from 5)
- **Chunk Size**: 1500 characters (increased from 1000)
- **Chunk Overlap**: 300 characters (increased from 200)
- **Terminal Display**: Full retrieved chunks are logged to terminal for debugging

## File Structure

```
context_retrieval_persona/
├── README.md                      # This file
├── context_retrieval_persona.py   # Main persona class
├── rag_core.py                    # Core RAG system
├── rag_integration_tool.py        # Agno tool wrapper
├── file_reader_tool.py            # Notebook content extraction
├── setup_rag_system.py           # Setup script
├── ynotebook_wrapper.py           # Jupyter notebook integration
├── test_context_retrieval.ipynb   # Test notebook
├── repo_context.md               # Generated markdown reports
├── PythonDataScienceHandbook/     # Cloned repository
│   └── notebooks/                 # 100+ handbook notebooks
└── vector_stores/                 # ChromaDB vector storage
    └── python_ds_handbook/
        ├── chroma.sqlite3
        └── metadata.json
```

## Performance Notes

- **First Run**: 5-10 minutes to build vector store
- **Subsequent Runs**: <5 seconds using cached vectors
- **Memory Usage**: ~500MB for full vector store
- **Search Speed**: <1 second for semantic queries

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install chromadb sentence-transformers langchain
   ```

2. **Vector Store Issues**: Force rebuild if corrupted
   ```python
   rag = create_handbook_rag(force_rebuild=True)
   ```

3. **Repository Problems**: Check git connectivity
   ```bash
   git clone https://github.com/jakevdp/PythonDataScienceHandbook.git
   ```

### Debug Information
```python
from rag_integration_tool import create_simple_rag_tools

rag_tool = create_simple_rag_tools()
status = rag_tool.get_system_status()
print(status)  # Detailed system diagnostics
```

## Contributing

To extend the system:

1. **Add New Search Methods**: Extend `RAGSearchTool` in `rag_integration_tool.py`
2. **Enhance Context Extraction**: Modify `NotebookReaderTool` in `file_reader_tool.py`
3. **Improve Agent Instructions**: Update agent prompts in `context_retriever_persona.py`

## License

This project uses the Python Data Science Handbook, which is available under the MIT License. See the handbook repository for full license details.