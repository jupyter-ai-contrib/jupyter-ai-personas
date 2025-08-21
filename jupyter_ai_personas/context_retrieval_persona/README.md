# Context Retrieval Persona

## Overview

The Context Retrieval Persona analyzes your data science notebooks and finds relevant resources from the Python Data Science Handbook using RAG (Retrieval-Augmented Generation). It employs a three-agent system to provide comprehensive analysis and actionable recommendations.

## Features

- **Intelligent Notebook Analysis**: Extracts libraries, analysis stage, domain, and objectives from your notebooks
- **Full Notebook RAG Search**: Returns complete relevant notebooks instead of fragments for comprehensive context
- **Handbook-Only Search**: Avoids redundant searching by focusing on external handbook content only
- **Multi-Agent Coordination**: NotebookAnalyzer, KnowledgeSearcher, and MarkdownGenerator working together
- **Comprehensive Markdown Reports**: Detailed reports with code examples, explanations, and next steps
- **Optimized Search**: 1-2 complete notebooks per query with clean terminal logging
- **Automatic Report Generation**: Creates `repo_context.md` with comprehensive analysis

## Architecture

### Three-Agent System

1. **NotebookAnalyzer**: Extracts structured context from your notebook

   - Uses `extract_rag_context` tool to read notebook content
   - Identifies libraries (pandas, numpy, sklearn, matplotlib, etc.)
   - Determines analysis stage (data_loading, eda, preprocessing, modeling, evaluation, visualization)
   - Outputs structured JSON with path, libraries, stage, domain, and objectives

2. **KnowledgeSearcher**: Performs targeted handbook-only RAG searches

   - Generates 4-5 targeted search queries based on notebook analysis
   - Uses `search_handbook_only` to find relevant complete notebooks
   - Each search returns 1-2 most relevant notebooks (not fragments)
   - Provides comprehensive handbook content to MarkdownGenerator

3. **MarkdownGenerator**: Creates detailed markdown reports
   - Synthesizes notebook analysis with RAG search results
   - Includes substantial content from retrieved handbooks
   - Creates cross-references between user's work and handbook examples
   - Saves comprehensive reports as `repo_context.md`

## Core Components

### Context Retrieval Persona (`persona.py`)

- Main persona class orchestrating the three-agent system
- Handles Jupyter AI integration and message processing
- Initializes AWS Bedrock models and agent coordination
- Manages greeting detection and team workflow

### RAG Tool (`rag_tool.py`)

Core RAG system with two main classes:

- **RAG**: Loads handbook content into ChromaDB vectorstore using HuggingFace embeddings
- **RAGTool**: Agno toolkit providing `search_handbook_only()` function
- Returns complete notebooks (1-2 per search) instead of fragments
- Clean terminal logging showing retrieved notebook titles and stats

### Notebook Reader Tool (`file_reader_tool.py`)

- `NotebookReaderTool`: Provides `extract_rag_context` function
- Reads complete notebook content and metadata
- Extracts context for the NotebookAnalyzer agent

## Installation & Setup

### Prerequisites

Install the context retrieval persona with its dependencies:

```bash
pip install -e ".[context_retriever]"
```

This installs:

- `agno` - Multi-agent framework
- `boto3` - AWS Bedrock integration
- `langchain` & `langchain-core` & `langchain-community` - RAG framework
- `sentence-transformers` - Embedding models
- `chromadb` - Vector database
- `nbformat` - Jupyter notebook reading

### Setup Python Data Science Handbook

```bash
# Clone the handbook repository
cd jupyter_ai_personas/context_retrieval_persona/
git clone https://github.com/jakevdp/PythonDataScienceHandbook.git
```

### AWS Configuration

Configure AWS credentials for Bedrock access:

```bash
aws configure
# or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## Usage

### Basic Usage

In Jupyter AI chat, use the @ mention to activate the persona:

```
@ContextRetrievalPersona notebook: /path/to/your/notebook.ipynb
Analyze my machine learning workflow and find relevant handbook resources
```

### Workflow Example

1. **User Request**: Provides notebook path and description
2. **NotebookAnalyzer**: Reads and analyzes notebook content
3. **KnowledgeSearcher**: Performs 4-5 targeted searches in handbook
4. **MarkdownGenerator**: Creates comprehensive `repo_context.md` report

### Terminal Output

During processing, you'll see clean RAG search logs:

```
🔍 RAG SEARCH: 'sklearn RandomForest classification'
📚 Found 2 relevant notebooks:
  1. 05.08-Random-Forests.ipynb (15 cells, 12450 chars)
  2. 05.03-Hyperparameters-and-Model-Validation.ipynb (22 cells, 18920 chars)
```

### Generated Report Structure

The `repo_context.md` file includes:

- **Executive Summary**: Overview of findings and connections
- **Current Notebook Analysis**: Libraries, stage, domain, objectives from your notebook
- **Comprehensive Handbook Resources**: Full code examples and explanations from retrieved notebooks
- **Detailed Code Examples**: Complete implementations from handbook
- **Cross-References and Learning Paths**: Connections between your work and handbook content
- **Actionable Implementation Steps**: Specific next steps based on analysis

## Technical Details

### RAG Implementation

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB with persistent storage
- **Search Strategy**: Similarity search returning complete notebooks (not fragments)
- **Results per Search**: 2 most relevant complete notebooks
- **Cell-Based Chunking**: Uses notebook cells as natural document boundaries

### Optimizations

- **Handbook-Only Search**: Avoids redundant notebook content in RAG results
- **Complete Notebook Retrieval**: Returns full notebooks instead of fragments for better context
- **One-Time Loading**: Vector store loaded once per session with handbook_loaded flag
- **Clean Logging**: Minimal terminal output showing only essential search information
- **JSON Validation Fix**: Uses `capture_validation_error=None` to suppress nbformat warnings

## File Structure

```
context_retrieval_persona/
├── README.md                      # This documentation
├── persona.py                     # Main persona class with three-agent system
├── rag_tool.py                   # RAG and RAGTool classes for handbook search
├── file_reader_tool.py            # NotebookReaderTool for content extraction
├── __init__.py                    # Package initialization
├── repo_context.md               # Generated markdown reports
├── PythonDataScienceHandbook/     # Cloned handbook repository
│   └── notebooks/                 # 100+ handbook notebooks
└── vector_stores/                 # ChromaDB vector storage
    └── rag/                       # Renamed from simple_rag
        ├── chroma.sqlite3
        └── [vector files]
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all required packages

   ```bash
   pip install -e ".[context_retriever]"
   ```

2. **Handbook Not Found**: Clone the handbook repository

   ```bash
   cd jupyter_ai_personas/context_retrieval_persona/
   git clone https://github.com/jakevdp/PythonDataScienceHandbook.git
   ```

3. **AWS/Bedrock Issues**: Configure AWS credentials

   ```bash
   aws configure
   ```

4. **JSON Validation Warnings**: These are now suppressed with `capture_validation_error=None`

5. **Vector Store Loading**: First run builds the vector store (5-10 minutes), subsequent runs are fast

## Contributing

To extend the system:

1. **Enhance RAG Search**: Modify `RAGTool` class in `rag_tool.py`
2. **Improve Context Extraction**: Update `NotebookReaderTool` in `file_reader_tool.py`
3. **Refine Agent Instructions**: Update agent prompts in `persona.py`
4. **Add New Analysis Capabilities**: Extend the three-agent system workflow
