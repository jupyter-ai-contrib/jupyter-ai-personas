"""
rag_core.py

Core RAG system for Python Data Science Handbook notebooks.
Handles repository cloning, content extraction, embedding, and vector storage.
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

# Suppress HuggingFace tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nbformat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Updated imports for LangChain community packages
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PythonDSHandbookRAG:
    """Core RAG system for Python Data Science Handbook notebooks."""
    
    # Class-level cache for embeddings to avoid re-initialization
    _embeddings_cache = {}
    
    def __init__(
        self, 
        repo_url: str = "https://github.com/jakevdp/PythonDataScienceHandbook.git",
        local_repo_path: str = None,
        vector_store_path: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1500,
        chunk_overlap: int = 300
    ):
        self.repo_url = repo_url
        
        # Get the directory where this script is located for absolute paths
        script_dir = Path(__file__).parent.absolute()
        
        # Set default paths relative to the script directory (data_science_persona)
        if local_repo_path is None:
            local_repo_path = script_dir / "PythonDataScienceHandbook"
        else:
            local_repo_path = Path(local_repo_path)
            if not local_repo_path.is_absolute():
                local_repo_path = script_dir / local_repo_path
        
        if vector_store_path is None:
            vector_store_path = script_dir / "vector_stores" / "python_ds_handbook"
        else:
            vector_store_path = Path(vector_store_path)
            if not vector_store_path.is_absolute():
                vector_store_path = script_dir / vector_store_path
        
        self.local_repo_path = local_repo_path.resolve()
        self.notebooks_path = self.local_repo_path / "notebooks"
        self.vector_store_path = vector_store_path.resolve()
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Log paths for debugging
        logger.info(f"📁 Repository path: {self.local_repo_path}")
        logger.info(f"📦 Vector store path: {self.vector_store_path}")
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        self._embeddings_cache = {}  # Cache for embeddings by model name
        
        # Ensure directories exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
    def setup_repository(self, force_clone: bool = False) -> bool:
        """Clone or update the Python Data Science Handbook repository."""
        try:
            if self.local_repo_path.exists() and not force_clone:
                logger.info(f"Repository already exists at {self.local_repo_path}")
                # Skip git pull for faster loading (only pull if explicitly requested)
                if force_clone:
                    try:
                        subprocess.run(
                            ["git", "-C", str(self.local_repo_path), "pull"],
                            check=True, capture_output=True, text=True
                        )
                        logger.info("Repository updated successfully")
                    except subprocess.CalledProcessError:
                        logger.warning("Could not update repository, using existing version")
                else:
                    logger.info("Skipping repository update for faster loading")
                return True
            
            # Clone repository
            if self.local_repo_path.exists():
                shutil.rmtree(self.local_repo_path)
                
            logger.info(f"Cloning repository to {self.local_repo_path}")
            subprocess.run(
                ["git", "clone", self.repo_url, str(self.local_repo_path)],
                check=True, capture_output=True, text=True
            )
            
            # Verify notebooks directory exists
            if not self.notebooks_path.exists():
                logger.error(f"Notebooks directory not found at {self.notebooks_path}")
                return False
                
            logger.info("Repository setup completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Repository setup failed: {e}")
            return False
    
    def extract_notebook_content(self) -> List[Document]:
        """Extract content from all notebooks in the repository."""
        documents = []
        
        if not self.notebooks_path.exists():
            logger.error(f"Notebooks directory not found: {self.notebooks_path}")
            return documents
        
        notebook_files = list(self.notebooks_path.glob("*.ipynb"))
        logger.info(f"Found {len(notebook_files)} notebook files")
        
        for notebook_path in notebook_files:
            try:
                # Read notebook
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Extract content from each cell
                for cell_idx, cell in enumerate(nb.cells):
                    cell_content = cell.get('source', '').strip()
                    if not cell_content:
                        continue
                    
                    # Create document with rich metadata
                    doc = Document(
                        page_content=cell_content,
                        metadata={
                            'source': str(notebook_path.relative_to(self.local_repo_path)),
                            'notebook_name': notebook_path.stem,
                            'cell_index': cell_idx,
                            'cell_type': cell.get('cell_type', 'unknown'),
                            'file_path': str(notebook_path)
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"Extracted {len([c for c in nb.cells if c.get('source')])} cells from {notebook_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {notebook_path}: {e}")
                continue
        
        logger.info(f"Total documents extracted: {len(documents)}")
        self.documents = documents
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents
        chunked_docs = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_size'] = len(doc.page_content)
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def initialize_embeddings(self) -> bool:
        """Initialize HuggingFace embeddings with caching."""
        try:
            # Check if embeddings are already cached
            if self.embedding_model in self._embeddings_cache:
                logger.info(f"Using cached embeddings for model: {self.embedding_model}")
                self.embeddings = self._embeddings_cache[self.embedding_model]
                return True
            
            logger.info(f"Initializing embeddings with model: {self.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Cache the embeddings for future use
            self._embeddings_cache[self.embedding_model] = self.embeddings
            logger.info("Embeddings initialized and cached successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def build_vector_store(self, force_rebuild: bool = False) -> bool:
        """Build or load vector store."""
        # Check if vector store already exists and is recent
        if not force_rebuild and self._vector_store_exists():
            logger.info("✅ Using existing vector store (fast loading)")
            return self._load_existing_vector_store()
        
        # Build new vector store
        logger.info("🔨 Building new vector store (this may take 5-10 minutes)...")
        
        # Extract and chunk documents
        documents = self.extract_notebook_content()
        if not documents:
            logger.error("No documents extracted for vector store")
            return False
        
        chunked_docs = self.chunk_documents(documents)
        if not chunked_docs:
            logger.error("No chunks created for vector store")
            return False
        
        # Initialize embeddings
        if not self.initialize_embeddings():
            return False
        
        try:
            # Create vector store
            logger.info("Creating Chroma vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=chunked_docs,
                embedding=self.embeddings,
                persist_directory=str(self.vector_store_path),
                collection_name="python_ds_handbook"
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            
            # Save metadata
            self._save_vector_store_metadata(len(documents), len(chunked_docs))
            
            logger.info(f"Vector store built successfully with {len(chunked_docs)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            return False
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store files exist."""
        required_files = [
            self.vector_store_path / "chroma.sqlite3",
            self.vector_store_path / "metadata.json"
        ]
        
        return all(f.exists() for f in required_files)
    
    def _load_existing_vector_store(self) -> bool:
        """Load existing vector store."""
        try:
            logger.info("Loading existing vector store...")
            
            # Initialize embeddings
            if not self.initialize_embeddings():
                return False
            
            # Load vector store
            self.vectorstore = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embeddings,
                collection_name="python_ds_handbook"
            )
            
            # Load metadata
            metadata = self._load_vector_store_metadata()
            logger.info(f"Loaded vector store with {metadata.get('total_chunks', 'unknown')} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {e}")
            return False
    
    def _save_vector_store_metadata(self, doc_count: int, chunk_count: int):
        """Save metadata about the vector store."""
        metadata = {
            'created_at': str(pd.Timestamp.now()),
            'embedding_model': self.embedding_model,
            'total_documents': doc_count,
            'total_chunks': chunk_count,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'repo_url': self.repo_url
        }
        
        metadata_path = self.vector_store_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_vector_store_metadata(self) -> Dict[str, Any]:
        """Load vector store metadata."""
        metadata_path = self.vector_store_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def search(self, query: str, k: int = 8, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search the vector store for relevant content."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Perform similarity search
            if filter_dict:
                docs = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'unknown'),
                    'notebook_name': doc.metadata.get('notebook_name', 'unknown'),
                    'cell_type': doc.metadata.get('cell_type', 'unknown')
                }
                results.append(result)
                
                # Log detailed search result with full content
                logger.info(f"📚 Result {i}: {result['notebook_name']} ({result['cell_type']})")
                logger.info(f"   Source: {result['source']}")
                logger.info(f"   Content Length: {len(result['content'])} characters")
                logger.info(f"   Full Content: {result['content']}")
                logger.info(f"   {'-' * 50}")
            
            logger.info(f"🔍 Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 8) -> List[tuple]:
        """Search with similarity scores."""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            formatted_results = []
            
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'source': doc.metadata.get('source', 'unknown'),
                    'notebook_name': doc.metadata.get('notebook_name', 'unknown'),
                    'cell_type': doc.metadata.get('cell_type', 'unknown')
                }
                formatted_results.append((result, score))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search with scores failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            'repository_path': str(self.local_repo_path),
            'vector_store_path': str(self.vector_store_path),
            'repository_exists': self.local_repo_path.exists(),
            'vector_store_exists': self._vector_store_exists(),
            'embeddings_initialized': self.embeddings is not None,
            'vectorstore_initialized': self.vectorstore is not None
        }
        
        # Add metadata if available
        if self._vector_store_exists():
            metadata = self._load_vector_store_metadata()
            stats.update(metadata)
        
        return stats
    
    def initialize_full_system(self, force_rebuild: bool = False) -> bool:
        """Initialize the complete RAG system."""
        logger.info("Initializing Python Data Science Handbook RAG system...")
        
        # Step 1: Setup repository
        if not self.setup_repository():
            logger.error("Failed to setup repository")
            return False
        
        # Step 2: Build vector store
        if not self.build_vector_store(force_rebuild=force_rebuild):
            logger.error("Failed to build vector store")
            return False
        
        logger.info("RAG system initialization completed successfully!")
        return True


# Global instance cache for singleton behavior
_rag_instance_cache = {}

# Convenience function for quick setup
def create_handbook_rag(force_rebuild: bool = False) -> PythonDSHandbookRAG:
    """Create and initialize Python Data Science Handbook RAG system."""
    cache_key = "default"
    
    # Return cached instance if available and not forcing rebuild
    if not force_rebuild and cache_key in _rag_instance_cache:
        logger.info("🚀 Using cached RAG instance (instant loading)")
        return _rag_instance_cache[cache_key]
    
    # Create new instance
    rag = PythonDSHandbookRAG()
    
    if rag.initialize_full_system(force_rebuild=force_rebuild):
        # Cache the instance for future use
        _rag_instance_cache[cache_key] = rag
        return rag
    else:
        logger.error("Failed to initialize RAG system")
        return None


# Quick test function
def test_rag_system():
    """Test the RAG system with a simple query."""
    logger.info("Testing RAG system...")
    
    rag = create_handbook_rag()
    if not rag:
        logger.error("RAG system initialization failed")
        return False
    
    # Test search
    results = rag.search("pandas dataframe groupby", k=3)
    if results:
        logger.info(f"Test successful! Found {len(results)} results")
        for i, result in enumerate(results[:2]):
            logger.info(f"Result {i+1}: {result['source']} - {result['content'][:100]}...")
        return True
    else:
        logger.error("Test failed - no results found")
        return False


if __name__ == "__main__":
    test_rag_system()