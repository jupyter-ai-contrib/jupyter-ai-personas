from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from langchain_core.messages import HumanMessage
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools

session = boto3.Session()
def main():

    agent = Agent(
        instructions=[ "You are PR review analyst, capable of checking PR changes with the whole repository.",
            "Use your tools to verify code quality",
            "verify whether PR contain existing logic but implemented in a different ways.",
            "Change in function logic, but function calls are not properly handled. ",
            "Inefficient code but logically correct: improve Time Complexity, Space Complexity.",
            "Do not create any issues or pull requests unless explicitly asked to do so"
        ],
        tools=[ReasoningTools(add_instructions=True, think=True, analyze=True)],
        show_tool_calls=True,
        
        model=AwsBedrock(
                id= "anthropic.claude-3-5-sonnet-20241022-v2:0",
                session=session
            ),
    )

    response = agent.run(
            "hi"
            # stream=False,
            # stream_intermediate_steps=False,
            # show_full_reasoning=True,
            )
    print(response)


if __name__ == "__main__":
    main()


