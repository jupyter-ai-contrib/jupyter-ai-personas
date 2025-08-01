import os
import subprocess
import tempfile
import re
from agno.tools import Toolkit
from agno.utils.log import logger
from agno.agent import Agent
import sys

sys.path.append("../knowledge_graph")
from jupyter_ai_personas.knowledge_graph.code_analysis_tool import CodeAnalysisTool
from jupyter_ai_personas.knowledge_graph.schema_validator import SchemaValidator


class RepoAnalysisTools(Toolkit):
    def __init__(self, **kwargs):
        # Use environment variables for Neo4j credentials
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable must be set")

        self.code_tool = CodeAnalysisTool()
        self.schema_validator = SchemaValidator(neo4j_uri, (neo4j_user, neo4j_password))

        super().__init__(
            name="repo_analysis",
            tools=[
                self.get_schema_info,
                self.query_codebase,
            ],
            **kwargs,
        )

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
