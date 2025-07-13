import os
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from neo4j import GraphDatabase
import hashlib

class BulkCodeAnalyzer:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
    
    def analyze_folder(self, folder_path, clear_existing=False):
        """Analyze all supported files in a folder and add to knowledge graph"""
        if clear_existing:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("Cleared existing graph")
        
        # Supported file extensions
        supported_extensions = {'.py'}      #for 1st phase just py
        
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1]
                if file_ext in supported_extensions:
                    all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} supported files")
        
        with self.driver.session() as session:
            for file_path in all_files:
                print(f"Analyzing: {file_path}")
                try:
                    if file_path.endswith('.py'):
                        self._analyze_file(file_path, session)
                    else:
                        self._analyze_non_python_file(file_path, session)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    def _analyze_file(self, file_path, session):
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = self.parser.parse(bytes(code, 'utf8'))
        self._extract_code_elements(tree.root_node, session, file_path)
    
    def _extract_code_elements(self, node, session, file_path, current_class=None):
        if node.type == 'class_definition':
            class_name = node.child_by_field_name("name").text.decode('utf8')
            session.run(
                "MERGE (c:Class {name: $name}) SET c.file = $file",
                name=class_name, file=file_path
            )
            
            superclasses = node.child_by_field_name("superclasses")
            if superclasses:
                for child in superclasses.children:
                    if child.type == 'identifier':
                        parent = child.text.decode('utf8')
                        session.run("MERGE (parent:Class {name: $parent})", parent=parent)
                        session.run(
                            "MATCH (parent:Class {name: $parent}), (child:Class {name: $child}) "
                            "MERGE (child)-[:INHERITS_FROM]->(parent)",
                            parent=parent, child=class_name
                        )
            
            for child in node.children:
                self._extract_code_elements(child, session, file_path, class_name)
        
        elif node.type == 'function_definition':
            func_name = node.child_by_field_name("name").text.decode('utf8')
            func_code = node.text.decode('utf8', errors='ignore')
            
            params_node = node.child_by_field_name("parameters")
            params = []
            if params_node:
                for child in params_node.children:
                    if child.type == 'identifier':
                        params.append(child.text.decode('utf8'))
            
            code_hash = hashlib.md5(func_code.encode()).hexdigest()
            
            session.run(
                "MERGE (f:Function {name: $name, file: $file}) "
                "SET f.code = $code, f.code_hash = $hash, f.parameters = $params, f.line_start = $start, f.line_end = $end",
                name=func_name, file=file_path, code=func_code, hash=code_hash, params=params,
                start=node.start_point[0], end=node.end_point[0]
            )
            
            if current_class:
                session.run(
                    "MATCH (c:Class {name: $class_name}), (f:Function {name: $func_name, file: $file}) "
                    "MERGE (c)-[:CONTAINS]->(f)",
                    class_name=current_class, func_name=func_name, file=file_path
                )
            
            # Extract function calls
            self._extract_function_calls(node, session, func_name, file_path)
        
        else:
            for child in node.children:
                self._extract_code_elements(child, session, file_path, current_class)
    
    def _analyze_non_python_file(self, file_path, session):
        """Analyze non-Python files (basic content indexing)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a File node for non-Python files
            session.run(
                "MERGE (f:File {path: $path}) SET f.content = $content, f.size = $size, f.type = $type",
                path=file_path, 
                content=content[:5000],  # Limit content size
                size=len(content),
                type=os.path.splitext(file_path)[1]
            )
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Create File node without content
            session.run(
                "MERGE (f:File {path: $path}) SET f.error = $error, f.type = $type",
                path=file_path,
                error=str(e),
                type=os.path.splitext(file_path)[1]
            )
    
    def _extract_function_calls(self, func_node, session, caller_name, file_path):
        """Extract function calls from a function body"""
        def find_calls(node):
            calls = []
            if node.type == 'call':
                func_expr = node.child_by_field_name('function')
                if func_expr and func_expr.type == 'identifier':
                    called_func = func_expr.text.decode('utf8')
                    calls.append(called_func)
                elif func_expr and func_expr.type == 'attribute':
                    # Handle method calls like obj.method()
                    attr = func_expr.child_by_field_name('attribute')
                    if attr:
                        called_func = attr.text.decode('utf8')
                        calls.append(called_func)
            
            for child in node.children:
                calls.extend(find_calls(child))
            return calls
        
        called_functions = find_calls(func_node)
        
        for called_func in called_functions:
            # Create CALLS relationship
            session.run(
                "MATCH (caller:Function {name: $caller, file: $file}) "
                "MERGE (called:Function {name: $called}) "
                "MERGE (caller)-[:CALLS]->(called)",
                caller=caller_name, called=called_func, file=file_path
            )


# analyzer = BulkCodeAnalyzer("neo4j://127.0.0.1:7687", ("neo4j", "Bhavana@97"))
# analyzer.analyze_folder("source_code", clear_existing=True)