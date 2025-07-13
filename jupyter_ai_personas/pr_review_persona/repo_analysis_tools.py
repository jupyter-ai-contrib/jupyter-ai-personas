import os
import subprocess
import tempfile
import re
from agno.tools import Toolkit
from agno.utils.log import logger
from agno.agent import Agent
import sys
sys.path.append('../knowledge_graph')
from jupyter_ai_personas.knowledge_graph.code_analysis_tool import CodeAnalysisTool
from jupyter_ai_personas.knowledge_graph.schema_validator import SchemaValidator

class RepoAnalysisTools(Toolkit):
    def __init__(self, **kwargs):
        # Use environment variables for Neo4j credentials
        neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if not neo4j_password:
            raise ValueError('NEO4J_PASSWORD environment variable must be set')
            
        self.code_tool = CodeAnalysisTool()
        self.schema_validator = SchemaValidator(neo4j_uri, (neo4j_user, neo4j_password))
        
        super().__init__(name="repo_analysis", tools=[
            self.get_schema_info,
            self.query_codebase,
            # self.get_function_source
            # self.find_class_relationships,
            # self.find_impact_analysis,
            # self.check_dependents_handled,
            # self.debug_database_contents,
            # self.get_nodes_by_file,
            # self.find_related_context,
            # self.get_nodes_by_lines,
            # self.analyze_signature_changes,
            # self.detect_semantic_patterns,
            # self.comprehensive_pr_analysis
        ], **kwargs)


    
    def get_schema_info(self, agent: Agent) -> str:
        """
        Get the knowledge graph schema information.
        
        Returns:
            str: Schema information for query writing
        """
        try:
            return self.schema_validator.generate_schema_info()
        except Exception as e:
            return f"Error getting schema: {str(e)}"
            


    def query_codebase(self, agent: Agent, query: str) -> str:
        """
        Execute a custom query on the analyzed codebase knowledge graph.
        
        Args:
            agent (Agent): The agent instance
            query (str): Cypher query to execute on the knowledge graph
            
        Returns:
            str: Query results
        """
        import time
        start_time = time.time()
        
        try:
            print(f"\n=== KG QUERY DEBUG ===")
            print(f"Full Cypher Query:")
            print(f"{query}")
            print(f"--- Executing Query ---")
            
            result = self.code_tool.query_code(query)
            query_time = time.time() - start_time
            
            print(f"Query Time: {query_time:.3f}s")
            print(f"Result Preview: {str(result)[:200]}...")
            print(f"=== END KG QUERY DEBUG ===\n")
            
            return result
        except Exception as e:
            print(f"KG Query Error: {str(e)}")
            print(f"=== END KG QUERY DEBUG ===\n")
            return f"Error executing query: {str(e)}"

    # def get_function_source(self, agent: Agent, function_name: str, class_name: str = None) -> str:
    #     """
    #     Get the source code of a specific function from the analyzed repository.
        
    #     Args:
    #         agent (Agent): The agent instance
    #         function_name (str): Name of the function to retrieve
    #         class_name (str, optional): Name of the class containing the function
            
    #     Returns:
    #         str: Source code of the function
    #     """
    #     import time
    #     start_time = time.time()
        
    #     try:
    #         result = self.code_tool.get_function_code(function_name, class_name)
    #         query_time = time.time() - start_time
    #         print(f"KG Function Lookup - Function: '{function_name}' | Class: '{class_name}' | Time: {query_time:.3f}s")
    #         return result
    #     except Exception as e:
    #         return f"Error retrieving function source: {str(e)}"

    # def find_class_relationships(self, agent: Agent, class_name: str) -> str:
    #     """
    #     Find inheritance relationships for a given class.
        
    #     Args:
    #         agent (Agent): The agent instance
    #         class_name (str): Name of the class to analyze
            
    #     Returns:
    #         str: Information about class relationships and structure
    #     """
    #     import time
    #     start_time = time.time()
        
    #     try:
    #         class_info = self.code_tool.get_class_info(class_name)
    #         related_classes = self.code_tool.find_related_classes(class_name)
    #         query_time = time.time() - start_time
    #         print(f"KG Class Analysis - Class: '{class_name}' | Time: {query_time:.3f}s")
    #         return f"{class_info}\n\n{related_classes}"
    #     except Exception as e:
    #         return f"Error analyzing class relationships: {str(e)}"
    
    # def find_impact_analysis(self, agent: Agent, target_name: str, target_type: str = "Function") -> str:
    #     """
    #     Find all modules/functions that would break if target is removed.
        
    #     Args:
    #         agent (Agent): The agent instance
    #         target_name (str): Name of function/class to analyze
    #         target_type (str): "Function" or "Class"
            
    #     Returns:
    #         str: Impact analysis results
    #     """
    #     import time
    #     start_time = time.time()
        
    #     try:
    #         if target_type == "Function":
    #             query = f"""
    #             MATCH (dependent:Function)-[:CALLS*]->(target:Function {{name: '{target_name}'}})
    #             RETURN DISTINCT dependent.file as affected_file, dependent.name as affected_function
    #             ORDER BY affected_file
    #             """
    #         else:  # Class
    #             query = f"""
    #             MATCH (child:Class)-[:INHERITS_FROM*]->(target:Class {{name: '{target_name}'}})
    #             OPTIONAL MATCH (child)-[:CONTAINS]->(f:Function)
    #             OPTIONAL MATCH (target)-[:CONTAINS]->(parent_method:Function)
    #             OPTIONAL MATCH (child)-[:CONTAINS]->(override:Function) WHERE override.name = parent_method.name
    #             RETURN DISTINCT child.file as affected_file, child.name as affected_class, 
    #                    f.name as child_method, parent_method.name as parent_method, 
    #                    override.name as potential_override
    #             ORDER BY affected_file
    #             """
            
    #         result = self.code_tool.query_code(query)
    #         query_time = time.time() - start_time
    #         print(f"KG Impact Analysis - Target: '{target_name}' | Type: '{target_type}' | Time: {query_time:.3f}s")
    #         return f"Impact Analysis for {target_type} '{target_name}':\n{result}"
    #     except Exception as e:
    #         return f"Error in impact analysis: {str(e)}"
    
    # def check_dependents_handled(self, agent: Agent, pr_diff: str, target_name: str, target_type: str = "Function") -> str:
    #     """Check if PR addresses dependent functions/classes that would be affected by changes"""
    #     import time
    #     start_time = time.time()
        
    #     try:
    #         # Get dependents from KG
    #         if target_type == "Function":
    #             query = f"""
    #             MATCH (dependent:Function)-[:CALLS]->(target:Function {{name: '{target_name}'}})
    #             RETURN DISTINCT dependent.name as name
    #             """
    #         else:  # Class
    #             query = f"""
    #             MATCH (child:Class)-[:INHERITS_FROM]->(target:Class {{name: '{target_name}'}})
    #             RETURN DISTINCT child.name as name
    #             """
            
    #         dependents_result = self.code_tool.query_code(query)
    #         modified_functions = self._extract_modified_functions(pr_diff)
            
    #         handled = []
    #         unhandled = []
            
    #         if isinstance(dependents_result, list):
    #             for dep in dependents_result:
    #                 dep_name = dep.get('name', '')
    #                 if dep_name in modified_functions:
    #                     handled.append(dep_name)
    #                 else:
    #                     unhandled.append(dep_name)
            
    #         query_time = time.time() - start_time
    #         print(f"KG Dependency Check - Target: '{target_name}' | Time: {query_time:.3f}s")
            
    #         result = f"Dependency Check for {target_type} '{target_name}':\n"
    #         if handled:
    #             result += f"✅ HANDLED in PR: {', '.join(handled)}\n"
    #         if unhandled:
    #             result += f"❌ NOT HANDLED in PR: {', '.join(unhandled)}\n"
    #             result += f"⚠️ ACTION REQUIRED: Update these dependents or ensure backward compatibility\n"
    #         if not handled and not unhandled:
    #             result += "ℹ️ No direct dependents found\n"
            
    #         return result
            
    #     except Exception as e:
    #         return f"Error checking dependents: {str(e)}"
    
    # def analyze_signature_changes(self, agent: Agent, pr_diff: str) -> str:
    #     """Detect function signature changes and type compatibility"""
    #     signature_changes = []
        
    #     old_sigs = re.findall(r'^-\s*def\s+(\w+)\s*\(([^)]*)\)', pr_diff, re.MULTILINE)
    #     new_sigs = re.findall(r'^\+\s*def\s+(\w+)\s*\(([^)]*)\)', pr_diff, re.MULTILINE)
        
    #     for func_name, old_params in old_sigs:
    #         for new_func, new_params in new_sigs:
    #             if func_name == new_func and old_params != new_params:
    #                 impact = self.find_impact_analysis(agent, func_name, "Function")
    #                 signature_changes.append(f"⚠️ Signature change: {func_name}({old_params}) → {func_name}({new_params})\n{impact}")
        
    #     return "\n".join(signature_changes) if signature_changes else "ℹ️ No signature changes detected"

    # def detect_semantic_patterns(self, agent: Agent, pr_diff: str) -> str:
    #     """Detect semantic patterns and potential issues"""
    #     patterns = []
        
    #     if re.search(r'password|token|secret|key', pr_diff, re.IGNORECASE):
    #         patterns.append("🔒 Security-sensitive code detected - review for credential exposure")
        
    #     if re.search(r'\.save\(\)|\.delete\(\)|\.update\(', pr_diff):
    #         patterns.append("💾 Database operations detected - verify transaction handling")
        
    #     if re.search(r'raise\s+\w+|except\s+\w+', pr_diff):
    #         patterns.append("⚠️ Exception handling modified - verify error propagation")
            
    #     if re.search(r'@.*auth|@.*login|@.*permission', pr_diff, re.IGNORECASE):
    #         patterns.append("🔐 Authorization decorators detected - verify access control")
        
    #     return "\n".join(patterns) if patterns else "ℹ️ No semantic risk patterns detected"

    # def comprehensive_pr_analysis(self, agent: Agent, pr_diff: str) -> str:
    #     """Complete PR analysis using multiple techniques"""
    #     analysis_results = []
        
    #     modified_functions = self._extract_modified_functions(pr_diff)
        
    #     analysis_results.append("=== COMPREHENSIVE PR ANALYSIS ===")
    #     analysis_results.append(f"Modified functions/classes: {', '.join(modified_functions)}")
        
    #     analysis_results.append("\n=== DEPENDENCY ANALYSIS ===")
    #     for func in modified_functions[:3]:  # Limit to first 3 for performance
    #         dep_check = self.check_dependents_handled(agent, pr_diff, func, "Function")
    #         analysis_results.append(dep_check)
        
    #     analysis_results.append("\n=== SIGNATURE ANALYSIS ===")
    #     analysis_results.append(self.analyze_signature_changes(agent, pr_diff))
        
    #     analysis_results.append("\n=== SEMANTIC PATTERNS ===")
    #     analysis_results.append(self.detect_semantic_patterns(agent, pr_diff))
        
    #     return "\n".join(analysis_results)

    # def _extract_modified_functions(self, pr_diff: str) -> list:
    #     """Extract function names from PR diff"""
    #     modified_functions = []
        
    #     function_pattern = r'^[+-]\s*def\s+(\w+)\s*\('
    #     matches = re.findall(function_pattern, pr_diff, re.MULTILINE)
    #     modified_functions.extend(matches)
        
    #     class_pattern = r'^[+-]\s*class\s+(\w+)\s*[\(:]'
    #     matches = re.findall(class_pattern, pr_diff, re.MULTILINE)
    #     modified_functions.extend(matches)
        
    #     return list(set(modified_functions))
    
    # def debug_database_contents(self, agent: Agent) -> str:
    #     """
    #     Debug what's actually in the database.
        
    #     Returns:
    #         str: Database contents summary
    #     """
    #     try:
    #         # Check all node types and their properties
    #         query = """
    #         MATCH (n)
    #         RETURN labels(n) as node_labels, 
    #                CASE WHEN n.file IS NOT NULL THEN n.file ELSE n.path END as file_path,
    #                CASE WHEN n.name IS NOT NULL THEN n.name ELSE 'no_name' END as name,
    #                CASE WHEN n.type IS NOT NULL THEN n.type ELSE 'no_type' END as type
    #         LIMIT 20
    #         """
            
    #         result = self.code_tool.query_code(query)
    #         return f"Database Contents Debug:\n{result}"
    #     except Exception as e:
    #         return f"Error debugging database: {str(e)}"
        

    # def get_nodes_by_file(self, agent: Agent, file_path: str) -> str:
    #     """Get all classes/functions in a specific file"""
    #     try:
    #         query = f"MATCH (n) WHERE n.file CONTAINS '{file_path}' RETURN n.name, labels(n)[0] as type"
    #         result = self.code_tool.query_code(query)
    #         return f"Nodes in file '{file_path}':\n{result}"
    #     except Exception as e:
    #         return f"Error getting nodes by file: {str(e)}"
        

    # def find_related_context(self, agent: Agent, class_name: str) -> str:
    #     """Get full context: class + parents + children + methods"""
    #     try:
    #         query = f"""
    #         MATCH (c:Class {{name: '{class_name}'}})
    #         OPTIONAL MATCH (c)-[:INHERITS_FROM]->(parent)
    #         OPTIONAL MATCH (child)-[:INHERITS_FROM]->(c)
    #         OPTIONAL MATCH (c)-[:CONTAINS]->(method)
    #         RETURN c.name, parent.name as parent, child.name as child, method.name as method
    #         """
    #         result = self.code_tool.query_code(query)
    #         return f"Related context for class '{class_name}':\n{result}"
    #     except Exception as e:
    #         return f"Error finding related context: {str(e)}"

    # def get_nodes_by_lines(self, agent: Agent, file_path: str, start_line: int, end_line: int) -> str:
    #     """Find nodes within specific line range"""
    #     try:
    #         query = f"""
    #         MATCH (n) WHERE n.file = '{file_path}' AND 
    #         n.line_start <= {end_line} AND n.line_end >= {start_line}
    #         RETURN n.name, labels(n)[0] as type
    #         """
    #         result = self.code_tool.query_code(query)
    #         return f"Nodes in '{file_path}' lines {start_line}-{end_line}:\n{result}"
    #     except Exception as e:
    #         return f"Error getting nodes by lines: {str(e)}"


