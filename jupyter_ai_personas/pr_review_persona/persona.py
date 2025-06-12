import os
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools
from langchain_core.messages import HumanMessage
from agno.tools.python import PythonTools
from agno.team.team import Team
from .ci_tools import CITools
from .template import PRPersonaVariables, PR_PROMPT_TEMPLATE

session = boto3.Session()

class PR_ReviewPersona(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="PR_ReviewPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized assistant for reviewing pull requests and providing detailed feedback.",
            system_prompt="You are a PR reviewer assistant that helps analyze code changes, provide feedback, and ensure code quality.",
        )
    

    def initialize_team(self, system_prompt):
        model_id = self.config.lm_provider_params["model_id"]
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is not set. Please set it with a plain GitHub personal access token (not GitHub Actions syntax).")

        code_quality = Agent(name="code_quality",
            role="Code Quality Analyst",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            markdown=True,
            instructions=[ "use GithubTools to get PR data"
                "Analyze code changes for quality and best practices:",
                "1. Review code style and consistency",
                "2. Check for code smells and anti-patterns using PylintTools" ,
                "3. Evaluate code complexity and readability",
                "4. Assess performance implications",
                "5. Verify error handling and edge cases",
                "6. Analyze CI failures:",
                "   - Use CITools to fetch and analyze CI failure data",
                "   - Review failure logs to identify issues",
                "   - Provide code recommendations based on failures",
                "   - You can fetch CI logs from GitHub and store them using fetch_ci_failure_data function",
                "   - use get_ci_logs to access the logs from memory.",
                system_prompt
            ],
            tools=[
                PythonTools(),
                # PRTools(),
                GithubTools( get_pull_requests= True, get_pull_request_changes= True, create_pull_request_comment= True ),
                CITools(),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        documentation_checker = Agent(name="documentation_checker",
            role="Documentation Specialist",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Review documentation completeness and quality:",
                "1. Verify docstrings for new/modified functions and classes",
                "2. Check README updates for new features or changes",
                "3. Verify return value documentation",
                "4. Check for documentation consistency",
            ],
            tools=[PythonTools()],
            markdown=True
        )

        security_checker = Agent(name="security_checker",
            role="Security Analyst",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Perform security analysis of code changes:",
                "1. Check for exposed sensitive information (API keys, tokens, credentials)",
                "2. Identify potential SQL injection vulnerabilities",
                "3. Verify proper input sanitization",
                "4. Check for insecure direct object references",
            ],
            tools=[PythonTools(), ReasoningTools(add_instructions=True, think=True, analyze=True,)],
            markdown=True
        )

        gitHub = Agent(name="gitHub",
            role="GitHub Specialist",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes",
                "Fetch and process pull request data",
                "Analyze code changes and provide structured feedback",
                "Create a comment on a specific line of a specific file in a pull request.",
                "Note: Requires a valid GitHub personal access token in GITHUB_TOKEN environment variable"
            ],
            tools=[
                GithubTools( create_pull_request_comment= True, get_pull_requests= True, get_pull_request_changes= True),
                # PRTools()
            ],
            markdown=True
        )


        pr_review_team = Team(
            name="pr-review-team",
            mode="coordinate",
            members=[code_quality, documentation_checker, security_checker, gitHub],
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Coordinate PR review process with specialized team members:",
                "1. Code Quality Analyst reviews code structure and patterns",
                "2. Documentation Specialist ensures comprehensive documentation",
                "3. Security Analyst identifies potential vulnerabilities",
                "4. GitHub Specialist manages repository operations",
                "5. Synthesize findings into a comprehensive review",
                # "6. Provide inline comments for each finding.",
                "Chat history: " + system_prompt
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            tools=[
                GithubTools( create_pull_request_comment= True, get_pull_requests= True, get_pull_request_changes= True),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        return pr_review_team

    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

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
            context=history_text
        )
        
        system_prompt = PR_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[0].content
        team = self.initialize_team(system_prompt)
        
        response = team.run(message.body, 
                            stream=False,
                            stream_intermediate_steps=True,
                            show_full_reasoning=True,)

        response = response.content
        async def response_iterator():
            yield response
        
        await self.stream_message(response_iterator())