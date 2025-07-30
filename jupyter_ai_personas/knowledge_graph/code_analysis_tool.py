from agno.tools import Toolkit
from .bulk_analyzer import BulkCodeAnalyzer
from neo4j import GraphDatabase
import ast
import os


class CodeAnalysisTool(Toolkit):
    def __init__(self):
        super().__init__(name="code_analysis")
        # Use environment variables for Neo4j credentials with defaults
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable must be set")

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.register(self.get_class_info)
        self.register(self.find_related_classes)
        self.register(self.query_code)
        self.register(self.get_function_code)

    def get_class_info(self, class_name: str) -> str:
        """Get detailed information about a class from the knowledge graph"""
        try:
            with self.driver.session() as session:
                # Get class info
                class_result = session.run(
                    "MATCH (c:Class {name: $class_name}) RETURN c.file as file",
                    class_name=class_name,
                )
                class_record = class_result.single()
                if not class_record:
                    return f"Class {class_name} not found in knowledge graph"

                inherit_result = session.run(
                    "MATCH (c:Class {name: $class_name})-[:INHERITS_FROM]->(parent:Class) "
                    "RETURN parent.name as parent_name",
                    class_name=class_name,
                )
                parents = [record["parent_name"] for record in inherit_result]

                method_result = session.run(
                    "MATCH (c:Class {name: $class_name})-[:CONTAINS]->(f:Function) "
                    "RETURN f.name as method_name, f.parameters as params",
                    class_name=class_name,
                )
                methods = [
                    (record["method_name"], record["params"])
                    for record in method_result
                ]

                info = f"Class {class_name}:\n"
                info += f"  File: {class_record['file']}\n"
                if parents:
                    info += f"  Inherits from: {', '.join(parents)}\n"
                info += f"  Methods:\n"
                for method_name, params in methods:
                    param_str = ", ".join(params) if params else ""
                    info += f"    {method_name}({param_str})\n"

                return info
        except Exception as e:
            return f"Error getting class info: {str(e)}"

    def get_function_code(self, function_name: str, class_name: str = None) -> str:
        """Get the source code of a function from the knowledge graph"""
        try:
            with self.driver.session() as session:
                # Query function with code directly
                if class_name:
                    result = session.run(
                        "MATCH (c:Class {name: $class_name})-[:CONTAINS]->(f:Function {name: $function_name}) "
                        "RETURN f.code as code, f.file as file, f.line_start as line_start, f.line_end as line_end",
                        class_name=class_name,
                        function_name=function_name,
                    )
                else:
                    result = session.run(
                        "MATCH (f:Function {name: $function_name}) "
                        "RETURN f.code as code, f.file as file, f.line_start as line_start, f.line_end as line_end",
                        function_name=function_name,
                    )

                record = result.single()
                if not record:
                    return f"Function {function_name} not found"

                # If code is stored directly on the function node
                if record["code"]:
                    return f"Function {function_name} code:\n{record['code']}"

        except Exception as e:
            return f"Error getting function code: {str(e)}"

    def find_related_classes(self, class_name: str) -> str:
        """Find all classes that inherit from the given class"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (related:Class)-[:INHERITS_FROM*]->(c:Class {name: $class_name}) "
                    "RETURN related.name as related_class",
                    class_name=class_name,
                )
                related = [record["related_class"] for record in result]
            if related:
                return f"Classes that inherit from {class_name}: {', '.join(related)}"
            else:
                return f"No classes inherit from {class_name}"
        except Exception as e:
            return f"Error finding related classes: {str(e)}"

    def query_code(self, query: str) -> str:
        """Execute custom Cypher queries on the code knowledge graph"""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = [dict(record) for record in result]
                return str(records) if records else "No results found"
        except Exception as e:
            return f"Query error: {str(e)}"