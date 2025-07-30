"""PRD Creation Agent for analyzing issues and creating Product Requirements Documents."""

import os
import re
import boto3
from typing import Optional
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.shell import ShellTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools

class PRDAgent:
    """Agent responsible for analyzing issues and creating PRDs."""
    
    def __init__(self, model_id: str, session):
        self.agent = Agent(
            name="prd_creator",
            role="Product Requirements Document Creator",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "CORE RESPONSIBILITY: Analyze issues and create comprehensive PRDs",
                
                "PRD STRUCTURE REQUIREMENTS:",
                "1. PROBLEM STATEMENT:",
                "   - Clear definition of the issue",
                "   - Impact and urgency assessment",
                "   - Affected stakeholders",
                
                "2. SOLUTION OVERVIEW:",
                "   - High-level approach",
                "   - Key components involved",
                "   - Technical considerations",
                
                "3. FUNCTIONAL REQUIREMENTS:",
                "   - Specific features needed",
                "   - User interactions",
                "   - System behaviors",
                
                "4. TECHNICAL REQUIREMENTS:",
                "   - Architecture considerations",
                "   - Performance requirements",
                "   - Security considerations",
                
                "5. ACCEPTANCE CRITERIA:",
                "   - Measurable success criteria",
                "   - Testing requirements",
                "   - Quality standards",
                
                "6. IMPLEMENTATION TASKS:",
                "   - Break down implementation into specific tasks",
                "   - Identify dependencies between tasks",
                "   - Prioritize tasks by importance",
                
                "ANALYSIS GUIDELINES:",
                "- Focus on minimal viable solution",
                "- Consider existing codebase patterns",
                "- Identify reusable components",
                "- Prioritize maintainability",
                "- Create actionable tasks that can be implemented"
            ],
            tools=[
                ShellTools(),
                FileTools(),
                GithubTools(get_issue=True, list_issue_comments=True),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )
    
    async def create_prd_from_issue(self, issue_url: str, repo_context: str = "") -> str:
        """Create a PRD from a GitHub issue URL using Agno agent."""
        # Extract issue details for context
        issue_match = re.search(r'github\.com/([^/]+/[^/]+)/issues/(\d+)', issue_url)
        if not issue_match:
            raise ValueError(f"Invalid GitHub issue URL: {issue_url}")
            
        repo_name, issue_number = issue_match.groups()
        
        # Use the agent with GithubTools to fetch issue content
        fetch_prompt = f"""Fetch GitHub issue {repo_name}#{issue_number} using get_issue with repo_path={repo_name} and issue_number={issue_number}.
        Provide the full issue title and description."""
        
        print(f"Fetching issue content for {repo_name}#{issue_number}...")
        
        try:
            # Run the agent to fetch issue details
            fetch_response = self.agent.run(fetch_prompt, stream=False)
            issue_content = fetch_response.content if hasattr(fetch_response, 'content') else str(fetch_response)
            print("Successfully fetched issue content")
        except Exception as e:
            print(f"Error fetching issue: {e}")
            issue_content = f"Issue URL: {issue_url}\nUnable to fetch details automatically."
        
        # Use the Agno agent to generate the PRD
        prompt = f"""
        Analyze the following GitHub issue and create a comprehensive Product Requirements Document (PRD):
        
        ISSUE CONTENT:
        {issue_content}
        
        REPOSITORY CONTEXT:
        {repo_context}
        
        Create a detailed PRD with the following sections:
        1. Issue Reference (Repository: {repo_name}, Issue: {issue_number})
        2. Problem Statement - Clear definition of the issue and its impact
        3. Solution Overview - High-level approach and key components
        4. Functional Requirements - Specific features and behaviors needed
        5. Technical Requirements - Architecture, performance, and security considerations
        6. Implementation Tasks - Break down implementation into specific tasks with dependencies
        7. Acceptance Criteria - Measurable success criteria
        
        For the Implementation Tasks section, create a list of specific tasks that can be directly used by TaskMaster.
        Each task should have:
        - A clear title
        - A detailed description
        - Priority (high/medium/low)
        - Dependencies on other tasks (if any)
        
        Focus on creating actionable requirements that can be broken down into specific tasks.
        Be specific and detailed about what needs to be implemented.
        """
        
        try:
            print("Generating PRD...")
            response = self.agent.run(prompt, stream=False)
            print("PRD generation completed successfully")
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error generating PRD: {e}")
            raise ValueError(f"Failed to generate PRD: {e}")
    
    # Alias for backward compatibility
    create_prd = create_prd_from_issue