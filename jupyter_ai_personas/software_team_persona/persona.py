from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from langchain_core.messages import HumanMessage
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools

from .template import SoftwareTeamVariables, _SOFTWARE_TEAM_PROMPT_TEMPLATE

session = boto3.Session()

class SoftwareTeamPersona(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="SoftwareTeamPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized software development team for Jupyter notebook cells with command-based functionality.",
            system_prompt="I am a software development team designed to help with coding tasks in Jupyter notebooks. I coordinate specialized team members: a planner who breaks down tasks into clear steps, a coder who implements solutions following best practices, a tester who ensures code quality through comprehensive testing, and a GitHub specialist who manages repository operations. Together, we can help you with planning, implementing, testing, and managing your code in GitHub.",
        )
    
    def initialize_team(self, system_prompt):
        model_id = self.config.lm_provider_params["model_id"]
        
        planner = Agent(name="planner",
            role="Strategic planner who breaks down tasks into clear, actionable steps",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Analyze user requests and break them down into clear, manageable steps",
                "Consider technical requirements, dependencies, and potential challenges"
            ],
            markdown=True, 
            show_tool_calls=True
        )

        coder = Agent(name="coder",
            role="Expert programmer responsible for implementing solutions",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Implement code following the planner's specifications",
                "Write clean, efficient, and well-documented code",
                "Follow Python best practices and PEP 8 style guidelines"
            ],
            tools=[PythonTools()],
            markdown=True, 
            show_tool_calls=True
            )

        tester = Agent(name="tester",
            role="Quality assurance engineer focused on testing and validation",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Write comprehensive unit tests for the implemented code",
                "Ensure test coverage for both normal cases and edge cases",
                "Include tests for error conditions and invalid inputs",
                "Follow testing best practices and naming conventions",
                "Verify that tests are independent and repeatable",
                "Document test cases and their purpose clearly",
                "Test both positive and negative scenarios"
            ],
            tools=[PythonTools()],
            markdown=True, 
            show_tool_calls=True
        )

        gitHub = Agent(name="gitHub",
            role="GitHub operations specialist managing repository interactions",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes",
                "Help with repository organization and maintenance",
                "Ensure proper Git workflow practices are followed",
                "Handle branch management and merging strategies",
                "Provide insights on repository metrics and activity patterns"
            ],
            tools=[GithubTools()],
            markdown=True, 
            show_tool_calls=True
        )

        fileManager = Agent(name="fileManager",
            role="File manager manages the local files, read and write.",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "Assist with local file management",
                "Only read a file when explicitly requested",
                "Only write to a file when explicitly requested"
            ],
            tools=[ FileTools()],
            markdown=True, 
            show_tool_calls=True
        )

        dev_team = Team(
            name="dev-team",
            mode="coordinate",
            members=[planner, coder, tester, gitHub, fileManager],
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=["Chat history is" + system_prompt,
                "Coordinate between planner, coder, tester, and GitHub specialist to deliver high-quality solutions",
                "Do not attempt to write test cases or test the code unless explicitly asked by user.",
                "Do not create new files unless explicitly asked by user."
                "Ensure smooth handoffs between planning, implementation, testing, and repository management phases",
                "Maintain clear communication between team members",
                "Validate that all requirements are met in the final solution",
                "Ensure code quality standards are maintained throughout the development process",
                "Address any conflicts or inconsistencies between different phases",
                "Facilitate collaboration through proper Git workflow and code review processes"
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=True
        )
        return dev_team

    async def process_message(self, message: Message):
        message_text = message.body

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

        variables = SoftwareTeamVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=history_text
        )

        system_prompt = _SOFTWARE_TEAM_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[0].content
        dev_team = self.initialize_team(system_prompt)

        response = dev_team.run(
            message_text,
            stream=False,
            stream_intermediate_steps=False,
            show_full_reasoning=True,
        )
        response = response.content

        async def response_iterator():
            yield response
        
        await self.stream_message(response_iterator())
