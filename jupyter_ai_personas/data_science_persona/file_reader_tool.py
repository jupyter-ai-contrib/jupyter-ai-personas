"""
File Reader Tool for retrieving complete notebook content.

This tool extracts all content from Jupyter notebooks including cells, 
outputs, and metadata to provide comprehensive context for analysis.
"""

import json
import os
from typing import Dict, Any, List, Optional
from agno.tools import Toolkit


class NotebookReaderTool(Toolkit):
    """Tool for reading and extracting complete content from Jupyter notebooks."""
    
    def __init__(self):
        super().__init__(name="notebook_reader")
        self.register(self.extract_rag_context)
    
    def extract_rag_context(self, notebook_path: str) -> str:
        """
        Extract complete content from a Jupyter notebook for RAG context.
        
        Args:
            notebook_path: Path to the .ipynb notebook file
            
        Returns:
            str: Formatted string containing all notebook content including cells,
                 outputs, markdown, and metadata
        """
        try:
            if not os.path.exists(notebook_path):
                return f"Error: Notebook file not found at {notebook_path}"
            
            if not notebook_path.endswith('.ipynb'):
                return f"Error: File must be a .ipynb notebook file, got {notebook_path}"
            
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Extract notebook metadata
            context = f"=== NOTEBOOK ANALYSIS ===\n"
            context += f"File: {notebook_path}\n"
            context += f"Kernel: {notebook.get('metadata', {}).get('kernelspec', {}).get('display_name', 'Unknown')}\n"
            context += f"Language: {notebook.get('metadata', {}).get('kernelspec', {}).get('language', 'Unknown')}\n\n"
            
            # Extract cells content
            cells = notebook.get('cells', [])
            context += f"=== NOTEBOOK CONTENT ({len(cells)} cells) ===\n\n"
            
            for i, cell in enumerate(cells, 1):
                cell_type = cell.get('cell_type', 'unknown')
                context += f"--- Cell {i} ({cell_type.upper()}) ---\n"
                
                # Get cell source
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = str(source)
                
                context += f"SOURCE:\n{source_text}\n"
                
                # Get cell outputs for code cells
                if cell_type == 'code':
                    outputs = cell.get('outputs', [])
                    if outputs:
                        context += f"OUTPUTS:\n"
                        for j, output in enumerate(outputs):
                            output_type = output.get('output_type', 'unknown')
                            context += f"  Output {j+1} ({output_type}):\n"
                            
                            # Handle different output types
                            if output_type == 'stream':
                                text = ''.join(output.get('text', []))
                                context += f"    {text}\n"
                            elif output_type == 'execute_result' or output_type == 'display_data':
                                data = output.get('data', {})
                                for mime_type, content in data.items():
                                    if mime_type == 'text/plain':
                                        if isinstance(content, list):
                                            content = ''.join(content)
                                        context += f"    {content}\n"
                                    elif mime_type == 'text/html':
                                        context += f"    [HTML OUTPUT]\n"
                                    elif 'image' in mime_type:
                                        context += f"    [IMAGE: {mime_type}]\n"
                            elif output_type == 'error':
                                ename = output.get('ename', 'Error')
                                evalue = output.get('evalue', '')
                                context += f"    ERROR: {ename}: {evalue}\n"
                
                context += "\n"
            
            # Extract imports and library usage
            imports = self._extract_imports(notebook)
            if imports:
                context += f"=== DETECTED LIBRARIES ===\n"
                for imp in imports:
                    context += f"- {imp}\n"
                context += "\n"
            
            # Extract data science context
            ds_context = self._extract_data_science_context(notebook)
            if ds_context:
                context += f"=== DATA SCIENCE CONTEXT ===\n{ds_context}\n"
            
            return context
            
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in notebook file {notebook_path}"
        except Exception as e:
            return f"Error reading notebook {notebook_path}: {str(e)}"
    
    def _extract_imports(self, notebook: Dict[str, Any]) -> List[str]:
        """Extract import statements from notebook cells."""
        imports = []
        cells = notebook.get('cells', [])
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = str(source)
                
                # Look for import statements
                lines = source_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        imports.append(line)
        
        return list(set(imports))  # Remove duplicates
    
    def _extract_data_science_context(self, notebook: Dict[str, Any]) -> str:
        """Extract data science context from notebook content."""
        context_items = []
        cells = notebook.get('cells', [])
        
        # Common data science patterns
        ds_patterns = {
            'pandas': ['pd.read_', 'DataFrame', '.head()', '.describe()', '.info()'],
            'numpy': ['np.array', 'np.mean', 'np.std', 'numpy'],
            'matplotlib': ['plt.', 'matplotlib', '.plot()', '.show()'],
            'seaborn': ['sns.', 'seaborn'],
            'sklearn': ['sklearn', 'fit()', 'predict()', 'score()'],
            'analysis': ['correlation', 'regression', 'classification', 'clustering'],
            'data_ops': ['merge', 'join', 'groupby', 'pivot', 'melt']
        }
        
        detected = {category: [] for category in ds_patterns.keys()}
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = str(source)
                
                for category, patterns in ds_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in source_text.lower():
                            detected[category].append(pattern)
        
        # Build context description
        active_categories = {k: list(set(v)) for k, v in detected.items() if v}
        
        if active_categories:
            context_items.append("Analysis stage indicators:")
            for category, patterns in active_categories.items():
                context_items.append(f"  {category}: {', '.join(patterns[:3])}")  # Limit to 3 examples
        
        return '\n'.join(context_items) if context_items else ""