import os
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import nbformat
from agno.tools import Toolkit

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        script_dir = Path(__file__).parent.absolute()
        self.handbook_path = script_dir / "PythonDataScienceHandbook" / "notebooks"
        self.persist_dir = script_dir / "vector_stores" / "rag"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        
    def load_content(self):
        """Load handbook content into vectorstore for similarity search."""
        documents = []
        
        for notebook_file in self.handbook_path.glob("*.ipynb"):
            with open(notebook_file, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT, capture_validation_error=None)
            
            for cell_idx, cell in enumerate(nb.cells):
                content = cell.get('source', '').strip()
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'source': notebook_file.name,
                            'type': 'handbook',
                            'cell_idx': cell_idx
                        }
                    ))
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        logger.info(f"Loaded {len(documents)} handbook cells")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """RAG similarity search - returns entire relevant notebooks."""
        docs = self.vectorstore.similarity_search(query, k=k*3)
        
        # Group by notebook file and get top notebooks
        notebook_scores = {}
        for doc in docs:
            source = doc.metadata['source']
            if source not in notebook_scores:
                notebook_scores[source] = 0
            notebook_scores[source] += 1  # Simple scoring by relevance count
        
        # Get top notebooks
        top_notebooks = sorted(notebook_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        results = []
        for notebook_name, _ in top_notebooks:
            # Load entire notebook
            notebook_path = self.handbook_path / notebook_name
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT, capture_validation_error=None)
            
            # Combine all cells into one result
            full_content = []
            for cell_idx, cell in enumerate(nb.cells):
                content = cell.get('source', '').strip()
                if content:
                    full_content.append(f"# Cell {cell_idx}\n{content}")
            
            results.append({
                'content': '\n\n'.join(full_content),
                'source': notebook_name,
                'type': 'full_notebook',
                'cell_count': len([c for c in nb.cells if c.get('source', '').strip()])
            })
        
        return results

class RAGTool(Toolkit):
    def __init__(self):
        super().__init__(name="rag")
        self.rag = None
        self.handbook_loaded = False
        
        self.register(self.search_handbook_only)
        
    def search_handbook_only(self, query: str, k: int = 5) -> str:
        """RAG similarity search in handbook only."""
        if not self.handbook_loaded:
            logger.info("Loading handbook (one-time initialization)")
            self.rag = RAG()
            self.rag.load_content()
            self.handbook_loaded = True
        
        results = self.rag.search(query, k=k)
        
        # Log RAG search results (titles only)
        print(f"\n🔍 RAG SEARCH: '{query}'")
        print(f"📚 Found {len(results)} relevant notebooks:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['source']} ({result['cell_count']} cells, {len(result['content'])} chars)")
        print("=" * 60)
        
        return json.dumps({
            "query": query,
            "total_results": len(results),
            "results": results
        }, indent=2)

def create_rag_tools() -> RAGTool:
    return RAGTool()