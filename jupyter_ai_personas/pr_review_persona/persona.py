import os
import re
import logging
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools
from langchain_core.messages import HumanMessage
from agno.team.team import Team
from .fetch_ci_failures import fetch_ci_failures
from .template import PRPersonaVariables, PR_PROMPT_TEMPLATE
from .pr_comment_tool import create_inline_pr_comments
from .repo_analysis_tools import RepoAnalysisTools
import subprocess
import tempfile
from jupyter_ai_personas.knowledge_graph.bulk_analyzer import BulkCodeAnalyzer
import sys

sys.path.append("../knowledge_graph")

session = boto3.Session()


class PRReviewPersona(BasePersona):
    # Heartbeat intervals
    FIRST_HEARTBEAT_DELAY = 120
    SECOND_HEARTBEAT_DELAY = 180
    THIRD_HEARTBEAT_DELAY = 300
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="PRReviewer",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized assistant for reviewing pull requests and providing detailed feedback.",
            system_prompt="You are a PR reviewer assistant that helps analyze code changes, provide feedback, and ensure code quality.",
        )

    def initialize_team(self, system_prompt):
        model_id = self.config_manager.lm_provider_params["model_id"]
        model_id = self.config_manager.lm_provider_params["model_id"]
        github_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not github_token:
            raise ValueError(
                "GITHUB_ACCESS_TOKEN environment variable is not set. Please set it with a plain GitHub personal access token (not GitHub Actions syntax)."
            )

        code_quality = Agent(
            name="code_quality",
            role="Code Quality Analyst",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "Review code quality and analyze CI failures:",
                "1. Get repository and PR information:",
                "   - Extract repo URL and PR number from the request",
                "   - Use GithubTools to fetch PR details",
                "2. ALWAYS check CI failures:",
                "   - MUST call fetch_ci_failures with repo_name and pr_number",
                "   - If failures found, analyze error messages and logs",
                "   - If no failures, mention that CI is passing",
                "   - Include CI status in your final report",
                "3 - MANDATORY KG Analysis for EVERY changed file:",
                "   - For EACH file in PR diff, you MUST run these KG queries:",
                "     a) query_codebase: MATCH (n) WHERE n.file CONTAINS 'filename' RETURN n.name, n.type",
                "     b) For each modified function: get_function_source(function_name)",
                "     c) For each modified class: find_class_relationships(class_name)",
                "   - For NEW files: MUST search for similar patterns with CONTAINS queries",
                "   - For MODIFIED files: MUST get current implementation before reviewing changes",
                "   - NEVER skip KG analysis - even if file seems simple",
                "4 Query Generation & Context Analysis (REQUIRED):",
                "   - FIRST: Describe the changes to the Query Generation Agent",
                "   - REQUEST: FOCUSED KG queries limited to PR scope",
                "   - EXECUTE: Only targeted queries, avoid system-wide searches",
                "   - CHECK DEPENDENCIES: Use check_dependents_handled for each modified function/class",
                "   - ANALYZE: Results within the context of actual changes",
                "   - The Query Agent will provide specialized Cypher queries based on:",
                "     * Change type (class/function/interface/utility)",
                "     * Risk level (high-impact vs isolated changes)",
                "     * Relationship patterns (inheritance/calls/dependencies)",
                "5. Review code quality:",
                "   - Code style and consistency",
                "   - Code smells and anti-patterns",
                "   - Complexity and readability",
                "   - Performance implications",
                "   - Error handling and edge cases",
                "4. MUST create inline comments for issues found:",
                "   - For each code issue, IMMEDIATELY call create_inline_pr_comments",
                "   - Use exact file paths from PR changes",
                "   - Use line numbers from the diff",
                "   - Do not just mention issues - CREATE the comments",
                "   - Use the exact format: [{\"path\": \"file.py\", \"position\": 10, \"body\": \"issue description\"}]",
            ],
            tools=[
                RepoAnalysisTools(),
                GithubTools(
                    get_pull_requests=True,
                    get_pull_request_changes=True,
                    get_file_content=True,
                    get_directory_content=True,
                ),
                fetch_ci_failures,
                create_inline_pr_comments,
                create_inline_pr_comments,
                ReasoningTools(add_instructions=True, think=True, analyze=True),
            ],
        )

        documentation_checker = Agent(
            name="documentation_checker",
            role="Documentation Specialist",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Review documentation completeness and quality:",
                "1. Verify docstrings for new/modified functions and classes",
                "2. Check README updates for new features or changes",
                "3. Verify return value documentation",
                "4. Check for documentation consistency",
            ],
            tools=[],
            markdown=True,
        )

        security_checker = Agent(
            name="security_checker",
            role="Security Analyst",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Perform security analysis of code changes:",
                "1. Check for exposed sensitive information (API keys, tokens, credentials)",
                "2. Identify potential SQL injection vulnerabilities",
                "3. Verify proper input sanitization",
                "4. Check for insecure direct object references",
            ],
            tools=[
                ReasoningTools(
                    add_instructions=True,
                    think=True,
                    analyze=True,
                ),
            ],
            markdown=True,
        )

        gitHub = Agent(
            name="github",
            role="GitHub Specialist",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes",
                "Fetch and process pull request data",
                "Analyze code changes and provide structured feedback",
                "Identify issues that need inline comments:",
                "   - Note specific code issues with file path and line number",
                "   - Report findings to the coordinator for comment posting",
                "Note: Requires a valid GitHub personal access token in GITHUB_ACCESS_TOKEN environment variable",
            ],
            tools=[
                GithubTools(
                    get_pull_requests=True,
                    get_pull_request_changes=True,
                ),
            ],
            markdown=True,
        )

        query_generator = Agent(
            name="query_generator",
            role="KG Query Specialist",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "You are a Neo4j Cypher query specialist for comprehensive PR analysis.",
                "KNOWLEDGE GRAPH SCHEMA:",
                "- Function nodes: {name, file, code, parameters, line_start, line_end, code_hash, embedding}",
                "- Class nodes: {name, file, embedding}",
                "- File nodes: {path, content, size, type, embedding}",
                "- Relationships: CALLS, INHERITS_FROM, CONTAINS",
                "- Embeddings: Use f.embedding for semantic similarity searches",
                "WORKFLOW:",
                "1. FIRST: Call get_schema_info() to understand the knowledge graph structure",
                "2. For each changed file/function/class, generate and execute ALL relevant queries:",
                "CRITICAL QUERY SCENARIOS TO COVER:",
                "A. Direct Dependencies:",
                "   - Who calls this function? MATCH (caller:Function)-[:CALLS]->(target:Function {name: 'X'}) RETURN caller.name, caller.file",
                "   - What does this function call? MATCH (f:Function {name: 'X'})-[:CALLS]->(called:Function) RETURN called.name, called.file",
                "B. Source Code Analysis:",
                "   - Get function source: MATCH (f:Function {name: 'X'}) RETURN f.name, f.code, f.parameters, f.line_start",
                "   - Compare signatures: MATCH (f:Function) WHERE f.file CONTAINS 'changed_file' RETURN f.name, f.parameters",
                "C. Cross-Module Impact:",
                "   - Cross-file dependencies: MATCH (f1:Function)-[:CALLS]->(f2:Function) WHERE f1.file <> f2.file AND f2.file CONTAINS 'changed_file' RETURN f1.file, f1.name, f2.name",
                "   - Module boundaries: Check if changes break module interfaces",
                "D. Inheritance Analysis:",
                "   - Child classes: MATCH (child:Class)-[:INHERITS_FROM]->(parent:Class {name: 'X'}) RETURN child.name, child.file",
                "   - Method overrides: MATCH (parent:Class)-[:CONTAINS]->(pm:Function), (child:Class)-[:INHERITS_FROM]->(parent), (child)-[:CONTAINS]->(cm:Function) WHERE pm.name = cm.name RETURN child.name, pm.name, pm.parameters, cm.parameters",
                "E. Breaking Change Detection:",
                "   - Dead code: MATCH (f:Function) WHERE NOT EXISTS((caller:Function)-[:CALLS]->(f)) AND f.file CONTAINS 'changed_file' RETURN f.name, f.file",
                "   - Circular dependencies: MATCH path = (f1:Function)-[:CALLS*2..5]->(f1) WHERE ANY(n IN nodes(path) WHERE n.file CONTAINS 'changed_file') RETURN [n IN nodes(path) | n.name]",
                "F. Test Coverage:",
                "   - Test relationships: MATCH (test:Function)-[:CALLS]->(f:Function) WHERE test.file CONTAINS 'test' AND f.file CONTAINS 'changed_file' RETURN test.name, f.name",
                "   - Missing tests: MATCH (f:Function) WHERE f.file CONTAINS 'changed_file' AND NOT EXISTS((test:Function)-[:CALLS]->(f) WHERE test.file CONTAINS 'test') RETURN f.name",
                "3. EXECUTE each query using query_codebase() and provide detailed analysis",
                "4. Include actual source code snippets when relevant using f.code property",
                "5. Use semantic search with embeddings: MATCH (f:Function) WHERE f.embedding IS NOT NULL",
                "6. Highlight potential breaking changes and risks",
            ],
            tools=[RepoAnalysisTools()],
            markdown=True,
        )

        pr_review_team = Team(
            name="pr-review-team",
            mode="coordinate",
            members=[
                query_generator,
                code_quality,
                documentation_checker,
                security_checker,
                gitHub,
            ],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Coordinate PR review process with specialized team members:",
                "1. Query Generator:",
                "   - WAIT for GitHub Specialist to provide ACTUAL PR diff data",
                "   - Generate queries ONLY based on real changes from diff",
                "   - NEVER generate queries based on assumptions",
                "   - Provide query recommendations to other team members",
                "2. Code Quality Analyst:",
                "   - Use query results from Query Generator for context",
                "   - Request specific source code analysis when needed",
                "   - Focus on code quality issues revealed by dependency analysis",
                "   - Check CI status and analyze any failures",
                "   - Identify breaking changes based on KG analysis",
                "3. Documentation Specialist:",
                "   - Keep analysis focused and concise",
                "   - Always create inline comments:",
                "2. Documentation Specialist:",
                "   - Review documentation completeness",
                "   - Focus on critical documentation issues",
                "4. Security Analyst:",
                "   - Check for security vulnerabilities",
                "   - Prioritize high-impact issues",
                "5. GitHub Specialist:",
                "   - FIRST ACTION: Call get_pull_request_changes() with actual repo URL and PR number",
                "   - VERIFY: Show actual PR diff data in response",
                "   - NEVER proceed without real GitHub data",
                "   - MUST run KG queries for each changed file from ACTUAL diff",
                "   - Provide deep code context using graph traversal and queries",
                "   - Keep PR metadata minimal",
                "6. CRITICAL - Code Quality Analyst handles inline comments:",
                "   - Code Quality Analyst will create inline comments for issues",
                "   - Other team members should report issues to Code Quality Analyst",
                "   - Focus on identifying and reporting actionable issues",
                "7. Synthesize findings:",
                "   - Combine key insights from all members",
                "   - Focus on actionable items",
                "   - Keep responses concise",
                "Chat history: " + system_prompt,
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=False,
            tools=[
                GithubTools(
                    get_pull_requests=True,
                    get_pull_request_changes=True,
                ),
                ReasoningTools(add_instructions=True, think=True, analyze=True),
            ],
        )

        return pr_review_team

    async def process_message(self, message: Message):

        # initial acknowledgment message
        pr_match = re.search(r'github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)', message.body)
        if pr_match:
            repo_name = pr_match.group(1)
            pr_number = pr_match.group(2)
            self.send_message(f"Got your request. Processing PR #{pr_number} from repo: {repo_name}")
 
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]

        history = YChatHistory(ychat=self.ychat, k=2)
        messages = await history.aget_messages()

        history_text = ""
        if messages:
            history_text = "\nPrevious conversation:\n"
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"

        variables = PRPersonaVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=history_text,
        )

        system_prompt = PR_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[
            0
        ].content

        # Analyze target repository for knowledge graph
        self._auto_analyze_repo(message.body)

        try:
            team = self.initialize_team(system_prompt)

            # Add periodic heartbeat messages during processing
            import asyncio

            # Flag to stop heartbeat when done
            processing = asyncio.Event()
            processing.set()

            async def heartbeat():
                heartbeat_steps = [
                    (self.FIRST_HEARTBEAT_DELAY, "⏳ Still processing large PR..."),
                    (self.SECOND_HEARTBEAT_DELAY, "⏳ Almost done..."),
                    (self.THIRD_HEARTBEAT_DELAY, "⏳ Taking longer than expected, please wait...")
                ]
                
                try:
                    for delay, message in heartbeat_steps:
                        await asyncio.sleep(delay)
                        if not processing.is_set():
                            break
                        self.send_message(message)
                except asyncio.CancelledError:
                    logger.debug("Heartbeat task cancelled")
                    raise  # Re-raise to properly terminate the task

            heartbeat_task = asyncio.create_task(heartbeat())

            try:
                response = await asyncio.to_thread(
                    team.run,
                    message.body,
                    stream=False,
                    stream_intermediate_steps=False,
                    show_full_reasoning=False,
                )

                # Stop heartbeat
                processing.clear()
                heartbeat_task.cancel()

                self.send_message(response.content)

            except Exception as run_error:
                processing.clear()
                heartbeat_task.cancel()
                raise run_error

        except ValueError as e:
            error_message = f"Configuration Error: {str(e)}\nThis may be due to missing or invalid environment variables, model configuration, or input parameters."
            self.send_message(error_message)

        except Exception as e:
            error_message = f"PR Review Error ({type(e).__name__}): {str(e)}\nAn unexpected error occurred while the PR review team was analyzing your request."
            self.send_message(error_message)

    def _auto_analyze_repo(self, pr_text: str):
        """Automatically extract repo URL and create knowledge graph"""
        patterns = [
            r"https://github\.com/([^/\s]+/[^/\s]+)",
            r"github\.com/([^/\s]+/[^/\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, pr_text)
            if match:
                repo_path = match.group(1).rstrip("/")
                repo_url = f"https://github.com/{repo_path}.git"
                self._clone_and_analyze(repo_url)
                break

    def _clone_and_analyze(self, repo_url: str):
        """Clone repository and create knowledge graph"""
        import time

        start_time = time.time()

        try:
            temp_dir = tempfile.mkdtemp()
            target_folder = os.path.join(temp_dir, "repo_analysis")

            clone_start = time.time()
            subprocess.run(
                ["git", "clone", repo_url, target_folder],
                check=True,
                capture_output=True,
            )
            clone_time = time.time() - clone_start

            kg_start = time.time()
            neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if not neo4j_password:
                raise ValueError("NEO4J_PASSWORD environment variable must be set")

            # embedding configuration
            embd_name = self.config_manager.em_provider.name
            embd_id = self.config_manager.em_provider_params["model_id"]

            analyzer = BulkCodeAnalyzer(
                neo4j_uri, (neo4j_user, neo4j_password), embd_name, embd_id
            )
            analyzer.analyze_folder(target_folder, clear_existing=True)
            kg_time = time.time() - kg_start

            total_time = time.time() - start_time
            print(
                f"KG Creation Times - Clone: {clone_time:.2f}s, Analysis: {kg_time:.2f}s, Total: {total_time:.2f}s"
            )

        except Exception as e:
            print(f"Error analyzing repository {repo_url}: {e}")
        finally:
            # Cleanup temporary directory
            import shutil

            if "temp_dir" in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
