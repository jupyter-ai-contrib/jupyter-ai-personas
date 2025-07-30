import os
import re
import tempfile
import subprocess
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from agno.tools.github import GithubTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.file import FileTools
from .enhanced_file_tools import EnhancedFileTools
from agno.tools.shell import ShellTools
from langchain_core.messages import HumanMessage
from agno.tools.python import PythonTools
from agno.team.team import Team
from .template import PRCreationPersonaVariables, PR_CREATION_PROMPT_TEMPLATE
from .repo_structure_tools import RepoStructureTools
import sys
sys.path.append('../knowledge_graph')
from jupyter_ai_personas.knowledge_graph.bulk_analyzer import BulkCodeAnalyzer
from jupyter_ai_personas.pr_review_persona.repo_analysis_tools import RepoAnalysisTools
from jupyter_ai_personas.task_master import TaskMasterClient, PRDAgent, Task
from jupyter_ai_personas.task_master.task_agent import TaskExecutionAgent

session = boto3.Session()

class PRCreationPersona(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_analyzer = None
        self.taskmaster = None
        self.prd_agent = None
        self.current_prd = None
        self.current_tasks = []
        self.current_repo_url = None
        self.current_issue_url = None
        self.local_repo_path = os.getenv("LOCAL_REPO_PATH", None)
        self.feature_branch = None
        self.folder_map = {}  # Maps component types to appropriate folders

    @property
    def defaults(self):
        return PersonaDefaults(
            name="PRCreationPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized assistant for analyzing issues and implementing code fixes with automated git operations.",
            system_prompt="You are a PR creation assistant that analyzes issues, designs solutions, and implements fixes with proper git workflow.",
        )
        
    def analyze_repository_structure(self, repo_path):
        """Analyze repository structure and create folder map for component placement."""
        if not repo_path or not os.path.exists(repo_path):
            print(f"Repository path does not exist: {repo_path}")
            return
            
        try:
            # Create RepoStructureTools instance
            structure_tools = RepoStructureTools()
            
            # Analyze folder structure
            print("Analyzing repository folder structure...")
            structure_analysis = structure_tools.analyze_folder_structure(None, repo_path)
            print(structure_analysis)
            
            # Get component placement map
            print("\nGenerating component placement map...")
            placement_map = structure_tools.get_component_placement_map(None, repo_path)
            print(placement_map)
            
            # Analyze project templates
            print("\nAnalyzing project templates...")
            template_analysis = structure_tools.analyze_project_templates(None, repo_path)
            print(template_analysis)
            
            # Parse the placement map to update the folder_map attribute
            for line in placement_map.split('\n'):
                if ': ' in line and line.startswith('- '):
                    comp_type, folder = line.replace('- ', '').split(': ', 1)
                    self.folder_map[comp_type.strip()] = folder.strip()
                    
            print(f"\nFolder map updated: {self.folder_map}")
            
            return {
                "structure_analysis": structure_analysis,
                "placement_map": placement_map,
                "template_analysis": template_analysis,
                "folder_map": self.folder_map
            }
            
        except Exception as e:
            print(f"Error analyzing repository structure: {str(e)}")
            return None

    def initialize_team(self, system_prompt):
        model_id = self.config_manager.lm_provider_params["model_id"]
        github_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_ACCESS_TOKEN environment variable is not set. Please set it with a plain GitHub personal access token.")

        # Issue Analysis Agent
        issue_analyzer = Agent(
            name="issue_analyzer",
            role="Issue Analysis Specialist",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "MANDATORY ISSUE ANALYSIS WORKFLOW - Follow these steps:",
                
                "STEP 1 - Parse Issue Requirements:",
                "   - Extract issue description and requirements",
                "   - Identify problem statement and expected behavior",
                "   - Determine scope and complexity",
                "   - List acceptance criteria",
                
                "STEP 2 - Repository Context Analysis:",
                "   - Use KG queries to understand codebase structure",
                "   - Identify relevant files and components",
                "   - Find similar patterns or existing implementations",
                "   - Analyze dependencies and relationships",
                
                "STEP 3 - Impact Assessment:",
                "   - Determine which files need modification",
                "   - Identify potential breaking changes",
                "   - Consider testing requirements",
                "   - Plan integration points",
                
                "OUTPUT: Structured analysis with clear requirements and affected components"
            ],
            tools=[RepoAnalysisTools(), ReasoningTools(add_instructions=True, think=True, analyze=True)]
        )

        # Architecture Designer Agent
        architect = Agent(
            name="architect",
            role="Solution Architecture Designer",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "MANDATORY ARCHITECTURE DESIGN WORKFLOW:",
                
                "STEP 1 - Solution Design:",
                "   - Design minimal solution architecture",
                "   - Plan file structure and organization",
                "   - Define interfaces and contracts",
                "   - Consider existing patterns and conventions",
                
                "STEP 2 - Repository Structure Analysis:",
                "   - Analyze existing folder structure patterns",
                "   - Identify where similar components are typically placed",
                "   - Create a folder map to guide file placement decisions",
                "   - Determine appropriate locations for new files",
                
                "STEP 3 - Implementation Strategy:",
                "   - Break down into implementable components",
                "   - Define clear separation of concerns",
                "   - Plan error handling and edge cases",
                "   - Consider performance implications",
                "   - SPECIFY EXACT FILE PATHS for all new/modified files",
                
                "STEP 4 - Integration Planning:",
                "   - Plan how new code integrates with existing",
                "   - Identify required imports and dependencies",
                "   - Consider backward compatibility",
                "   - Plan testing approach",
                
                "OUTPUT: Detailed implementation plan with file-by-file changes and EXACT file paths"
            ],
            tools=[
                RepoAnalysisTools(), 
                RepoStructureTools(),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        # Code Implementation Agent
        code_implementer = Agent(
            name="code_implementer",
            role="Code Implementation Specialist",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "MANDATORY CODE IMPLEMENTATION WORKFLOW:",
                
                "STEP 1 - Repository Setup:",
                f"   - {'Use existing repository at ' + self.local_repo_path if self.local_repo_path else 'Clone repository using shell commands'}",
                "   - If using existing repo: fetch latest changes",
                f"   - Use the feature branch: {self.feature_branch if self.feature_branch else 'feature/issue-description'}",
                "   - Check if branch exists: git branch --list <branch_name>",
                "   - If exists: git checkout <branch_name>",
                "   - If not exists: git checkout -b <branch_name> from main/master",
                "   - Verify current codebase state",
                
                "STEP 2 - Repository Structure Analysis:",
                "   - Analyze existing folder structure patterns",
                "   - Identify where similar components are typically placed",
                "   - Create a folder map to guide file placement decisions",
                "   - Determine appropriate locations for new files",
                
                "STEP 3 - Code Implementation:",
                "   - Write MINIMAL code that addresses the issue",
                "   - Follow existing code patterns and style",
                "   - Implement proper error handling",
                "   - Focus ONLY on the specific issue requirements",
                
                "STEP 4 - File Operations:",
                "   - Create/modify files using FileTools",
                "   - NEVER create files in the repository root",
                "   - Always follow project structure conventions",
                "   - Ensure parent directories exist before file creation",
                "   - Validate file paths against project patterns",
                
                "CRITICAL REQUIREMENTS:",
                "- Write ONLY the minimal code needed",
                "- Follow existing patterns exactly",
                "- Place files in appropriate directories",
                "- NO verbose implementations",
                "- Focus on the specific issue only"
            ],
            tools=[
                ShellTools(),
                EnhancedFileTools(),
                PythonTools(),
                RepoAnalysisTools(),
                RepoStructureTools(),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        # Git Operations Agent
        git_manager = Agent(
            name="git_manager",
            role="Git Operations Manager",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "MANDATORY GIT WORKFLOW:",
                
                "STEP 1 - Repository Operations:",
                "   - Use shell commands for git operations",
                f"   - {'Use existing repository at ' + self.local_repo_path if self.local_repo_path else 'Clone main branch from repository'}",
                "   - If using existing repo: fetch latest changes and checkout main branch",
                "   - Verify repository state and structure",
                
                "STEP 2 - Branch Management:",
                f"   - Use the feature branch: {self.feature_branch if self.feature_branch else 'feature/issue-description'}",
                "   - Check if branch exists: git branch --list <branch_name>",
                "   - If exists: git checkout <branch_name>",
                "   - If not exists: git checkout -b <branch_name>",
                
                "STEP 3 - Commit Operations:",
                "   - Stage files: git add .",
                "   - Create clear commit messages: git commit -m 'description'",
                "   - Follow conventional commit format if used in repo",
                
                "STEP 4 - Push Operations:",
                "   - Push feature branch: git push -u origin branch-name",
                "   - DO NOT create pull request (user will do manually)",
                
                "CRITICAL REQUIREMENTS:",
                "- NEVER create pull requests automatically",
                "- Always push to feature branch, never main",
                "- Use clear, descriptive commit messages"
            ],
            tools=[
                ShellTools(),
                GithubTools(get_pull_requests=True, get_pull_request_changes=True),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        # Configure EnhancedFileTools with repository structure information
        if self.local_repo_path and os.path.exists(self.local_repo_path):
            # Get the EnhancedFileTools instance from code_implementer
            for tool in code_implementer.tools:
                if isinstance(tool, EnhancedFileTools):
                    # Set repository path
                    tool.set_repo_path(self.local_repo_path)
                    # Set structure tools
                    tool.set_structure_tools(RepoStructureTools())
                    print(f"Configured EnhancedFileTools with repository path: {self.local_repo_path}")
                    break
        
        # Create the coordinating team
        pr_creation_team = Team(
            name="pr-creation-team",
            mode="coordinate",
            members=[issue_analyzer, architect, code_implementer, git_manager],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Coordinate PR creation process with clear separation of tasks:",
                
                "PHASE 1 - ANALYSIS (Issue Analyzer):",
                "   - Parse and understand the issue requirements",
                "   - Analyze repository context using KG queries",
                "   - Identify affected components and scope",
                "   - Provide structured analysis to team",
                
                "PHASE 2 - ARCHITECTURE (Architect):",
                "   - Wait for issue analysis completion",
                "   - Design minimal solution architecture",
                "   - Plan implementation strategy",
                "   - Create detailed file-by-file implementation plan",
                
                "PHASE 3 - IMPLEMENTATION (Code Implementer):",
                "   - Wait for architecture design completion",
                "   - Set up repository and create branch",
                "   - Implement code changes following the plan",
                "   - Write ONLY minimal code addressing the issue",
                "   - Ensure code follows existing patterns",
                
                "PHASE 4 - GIT OPERATIONS (Git Manager):",
                "   - Wait for code implementation completion",
                "   - Commit changes with clear messages",
                "   - Push feature branch to remote",
                "   - Provide branch information for manual PR creation",
                
                "COORDINATION RULES:",
                "- Each phase must complete before next begins",
                "- Share context and findings between agents",
                "- Maintain focus on minimal, targeted solutions",
                "- Ensure proper git workflow throughout",
                
                "Chat history: " + system_prompt
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            tools=[
                ShellTools(),
                FileTools(),
                GithubTools(get_pull_requests=True, get_pull_request_changes=True),
                RepoAnalysisTools(),
                ReasoningTools(add_instructions=True, think=True, analyze=True)
            ]
        )

        return pr_creation_team

    def _initialize_taskmaster(self):
        """Initialize the TaskMaster client and PRD agent if not already done."""
        if self.taskmaster is None:
            self.taskmaster = TaskMasterClient()
            
        if self.prd_agent is None:
            model_id = self.config_manager.lm_provider_params["model_id"]
            session = boto3.Session()
            self.prd_agent = PRDAgent(model_id=model_id, session=session)
    
    def _parse_command(self, message: str):
        """Parse user command to determine if it's a task-related command."""
        # Check for issue URL
        issue_match = re.search(r'(https://github\.com/[^/\s]+/[^/\s]+/issues/\d+)', message)
        if issue_match:
            return {
                "action": "process_issue",
                "issue_url": issue_match.group(1)
            }
            
        # Check for task details command
        task_details_match = re.search(r'(?:show|get|display)\s+task\s+(?:details|info)?\s+(?:for|of)?\s*[#]?(\d+)', message, re.IGNORECASE)
        if task_details_match:
            return {
                "action": "show_task_details",
                "task_id": task_details_match.group(1)
            }
            
        # Check for implement task command
        impl_task_match = re.search(r'implement\s+task\s*[#]?(\d+)', message, re.IGNORECASE)
        if impl_task_match:
            return {
                "action": "implement_task",
                "task_id": impl_task_match.group(1)
            }
            
        # Check for list tasks command
        if re.search(r'(?:list|show|get)\s+(?:all\s+)?tasks', message, re.IGNORECASE):
            return {
                "action": "list_tasks"
            }
            
        # Check for create PR command
        if re.search(r'(?:create|make)\s+(?:a\s+)?(?:PR|pull\s*request)(?:\s+for\s+(?:completed|done)\s+tasks)?', message, re.IGNORECASE):
            return {
                "action": "create_pr"
            }
            
        # Default to standard PR creation
        return {
            "action": "standard_pr_creation"
        }
    
    async def _process_issue(self, issue_url: str):
        """Process a GitHub issue to create PRD and tasks."""
        try:
            print(f"Processing issue: {issue_url}")
            self._initialize_taskmaster()
            
            # Extract repo URL from issue URL
            repo_match = re.search(r'(https://github\.com/[^/]+/[^/]+)/', issue_url)
            if repo_match:
                self.current_repo_url = repo_match.group(1) + ".git"
                print(f"Extracted repo URL: {self.current_repo_url}")
            
            self.current_issue_url = issue_url
            
            # Create a consistent feature branch name based on the issue number
            issue_number = issue_url.split('/')[-1]
            self.feature_branch = f"feature/issue-{issue_number}"
            print(f"Using feature branch: {self.feature_branch} for all tasks")
            
            # If LOCAL_REPO_PATH is set but doesn't exist, clone the repository there
            if self.local_repo_path and self.current_repo_url and not os.path.exists(self.local_repo_path):
                print(f"LOCAL_REPO_PATH is set to {self.local_repo_path} but doesn't exist. Will clone repository there.")
                # The actual cloning will happen in _validate_local_repo when called
            
            # Create PRD from issue
            print("Creating PRD from issue...")
            raw_prd = await self.prd_agent.create_prd_from_issue(issue_url)
            
            # Check for and remove any repetition in the PRD
            # This can happen if the model generates multiple PRDs for the same issue
            if "# Product Requirements Document" in raw_prd:
                # Find all occurrences of PRD headers
                prd_headers = [m.start() for m in re.finditer(r'# Product Requirements Document', raw_prd)]
                if len(prd_headers) > 1:
                    # Keep only the first PRD (up to the second header)
                    self.current_prd = raw_prd[:prd_headers[1]].strip()
                    print("Detected and removed duplicate PRD content")
                else:
                    self.current_prd = raw_prd
            else:
                self.current_prd = raw_prd
                
            print(f"PRD created successfully! Length: {len(self.current_prd)} chars")
            
            # Save PRD to file for debugging
            with open("generated_prd.md", "w") as f:
                f.write(self.current_prd)
            print("PRD saved to generated_prd.md")
            
            # Create tasks from PRD
            print("Creating tasks from PRD...")
            self.current_tasks = await self.taskmaster.create_tasks_from_prd(self.current_prd)
            print(f"Created {len(self.current_tasks)} tasks successfully!")
            
            # Auto-analyze repository
            print("Auto-analyzing repository...")
            self._auto_analyze_repo(issue_url)
            print("Repository analysis complete")
            
            # Get available tasks (no dependencies)
            available_tasks = self.taskmaster.get_available_tasks()
            print(f"Found {len(available_tasks)} available tasks with no dependencies")
            
            # Format response - SIMPLIFIED to show only PRD and available tasks
            response = f"## Issue Processed Successfully\n\n"
            response += f"Issue URL: {issue_url}\n\n"
            response += f"### PRD\n\n"
            response += f"{self.current_prd}\n\n"  # Show full PRD
            
            # Only show available tasks with no dependencies
            response += f"### Available Tasks\n"
            if available_tasks:
                response += f"These tasks have no dependencies and can be implemented immediately:\n\n"
                # Only show title and description for each task
                for task in available_tasks:
                    response += f"**Task #{task.id}: {task.title}**\n"
                    response += f"Description: {task.description}\n\n"
                
                # Add quick links to details and implementation
                response += f"\n\nCommands:\n"
                response += f"- 'show task details for #ID' to see implementation details of a specific task\n"
                response += f"- 'implement task #ID' to implement a specific task\n"
                response += f"- 'list tasks' to see all tasks\n"
            else:
                response += f"No tasks are currently ready for implementation.\n"
            
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _process_issue: {e}\n{error_trace}")
            return f"## Error Processing Issue\n\nAn error occurred while processing the issue: {str(e)}\n\nPlease try again or contact support."
    
    async def _show_task_details(self, task_id: str):
        """Show details for a specific task, including implementation details."""
        try:
            self._initialize_taskmaster()
            
            if not self.current_tasks:
                return "No tasks available. Please process an issue first."
            
            task = self.taskmaster.get_task_by_id(task_id)
            if not task:
                return f"Task with ID {task_id} not found."
            
            # Format the task with full details
            response = f"## Task #{task_id} Details\n\n"
            response += f"**{task.title}**\n\n"
            response += f"**Description:** {task.description}\n\n"
            response += f"**Priority:** {task.priority}\n"
            response += f"**Status:** {task.status}\n\n"
            
            # Show dependencies
            if task.dependencies:
                response += f"**Dependencies:**\n"
                for dep_id in task.dependencies:
                    dep_task = self.taskmaster.get_task_by_id(dep_id)
                    status = "✅ Completed" if dep_task and dep_task.status == "done" else "⏳ Pending"
                    response += f"- Task #{dep_id}: {status}\n"
                response += "\n"
            
            # Show implementation details
            if task.details:
                response += f"**Implementation Details:**\n```\n{task.details}\n```\n\n"
            
            # Show test strategy
            if task.test_strategy:
                response += f"**Test Strategy:**\n```\n{task.test_strategy}\n```\n\n"
            
            # Add implementation option if task is available
            available_tasks = self.taskmaster.get_available_tasks()
            if task in available_tasks:
                response += f"**This task has no unmet dependencies and can be implemented immediately.**\n"
                response += f"\nTo implement this task, type: 'implement task #{task_id}'\n"
            else:
                # Show dependencies that need to be completed first
                if task.dependencies:
                    response += f"\n**This task has dependencies that must be completed first:**\n"
                    for dep_id in task.dependencies:
                        dep_task = self.taskmaster.get_task_by_id(dep_id)
                        if dep_task and dep_task.status != "done":
                            response += f"- Task #{dep_id}: {dep_task.title} (Status: {dep_task.status})\n"
            
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _show_task_details: {e}\n{error_trace}")
            return f"## Error Showing Task Details\n\nAn error occurred while showing task details: {str(e)}\n\nPlease try again or contact support."
    
    async def _implement_task(self, task_id: str):
        """Implement a specific task using TaskExecutionAgent."""
        try:
            print(f"Implementing task #{task_id}...")
            self._initialize_taskmaster()
            
            if not self.current_tasks:
                return "No tasks available. Please process an issue first."
            
            task = self.taskmaster.get_task_by_id(task_id)
            if not task:
                return f"Task with ID {task_id} not found."
            
            # Check if task dependencies are met
            available_tasks = self.taskmaster.get_available_tasks()
            if task not in available_tasks:
                # Show which dependencies need to be completed
                unmet_deps = []
                for dep_id in task.dependencies:
                    dep_task = self.taskmaster.get_task_by_id(dep_id)
                    if dep_task and dep_task.status != "done":
                        unmet_deps.append(dep_id)
                
                response = f"## Task {task_id} Has Unmet Dependencies\n\n"
                response += f"The following dependencies must be completed first:\n\n"
                
                for dep_id in unmet_deps:
                    dep_task = self.taskmaster.get_task_by_id(dep_id)
                    if dep_task:
                        response += f"- Task #{dep_id}: {dep_task.title}\n"
                        response += f"  To implement: 'implement task #{dep_id}'\n"
                
                return response
            
            print(f"Task #{task_id} is available for implementation")
            
            # Mark task as in-progress
            self.taskmaster.update_task_status(task_id, "in-progress")
            
            # Create repository context information
            repo_context = f"Repository URL: {self.current_repo_url}\n"
            if self.feature_branch:
                repo_context += f"Feature Branch: {self.feature_branch}\n"
                repo_context += "Use this feature branch for all tasks.\n"
            if self.local_repo_path:
                # Make sure the path is absolute
                abs_path = os.path.abspath(self.local_repo_path)
                repo_context += f"Local Repository Path: {abs_path}\n"
                repo_context += f"IMPORTANT: Save all files to {abs_path}\n"
                repo_context += "Use the existing local repository instead of cloning a new one.\n"
                print(f"Using local repository path: {abs_path}")
                
                # Ensure the directory exists
                if not os.path.exists(abs_path):
                    os.makedirs(abs_path, exist_ok=True)
                    print(f"Created directory: {abs_path}")
                    
                # Clone the repository if it's not already a git repository
                if not os.path.exists(os.path.join(abs_path, '.git')):
                    if self.current_repo_url:
                        print(f"Cloning repository {self.current_repo_url} to {abs_path}")
                        try:
                            # Remove any existing content
                            if os.path.exists(abs_path) and os.listdir(abs_path):
                                print("Removing existing content before cloning")
                                for item in os.listdir(abs_path):
                                    item_path = os.path.join(abs_path, item)
                                    if os.path.isdir(item_path):
                                        import shutil
                                        shutil.rmtree(item_path)
                                    elif os.path.isfile(item_path):
                                        os.remove(item_path)
                            
                            # Clone the repository
                            subprocess.run(["git", "clone", self.current_repo_url, abs_path], check=True, capture_output=True)
                            print(f"Successfully cloned repository to {abs_path}")
                        except Exception as e:
                            print(f"Warning: Failed to clone repository: {e}")
                    else:
                        print(f"Warning: No repository URL available to clone. Initializing empty git repository.")
                        try:
                            subprocess.run(["git", "init"], cwd=abs_path, check=True, capture_output=True)
                        except Exception as e:
                            print(f"Warning: Failed to initialize git repository: {e}")
            
            # Add PRD context
            repo_context += f"\nPRD Context:\n{self.current_prd[:1000]}...\n"
            
            # Initialize TaskExecutionAgent
            model_id = self.config_manager.lm_provider_params["model_id"]
            task_agent = TaskExecutionAgent(model_id=model_id, session=session)
            
            # Execute the task
            print(f"Running TaskExecutionAgent for task #{task_id}")
            result = await task_agent.execute_task(task, repo_context)
            
            # Update task status to done
            print(f"Updating task #{task_id} status to done")
            # First try using the TaskMaster command directly
            try:
                work_dir = os.getcwd()
                print(f"Running: npx task-master set-status --status=done --id={task_id}")
                cmd_result = subprocess.run([
                    'npx', 'task-master', 'set-status',
                    f'--status=done',
                    f'--id={task_id}'
                ], cwd=work_dir, capture_output=True, text=True)
                
                if cmd_result.returncode == 0:
                    print(f"Successfully updated task {task_id} status to done via direct command")
                else:
                    print(f"Direct command failed: {cmd_result.stderr}")
                    # Fall back to using the TaskMasterClient
                    success = self.taskmaster.update_task_status(task_id, "done")
                    if not success:
                        print(f"Warning: Failed to update task status in TaskMaster. Updating in memory only.")
                        # Update the task status in memory
                        for t in self.current_tasks:
                            if t.id == task_id:
                                t.status = "done"
                                break
            except Exception as e:
                print(f"Error updating task status: {e}")
                # Fall back to using the TaskMasterClient
                self.taskmaster.update_task_status(task_id, "done")
            
            # Check if new tasks are now available
            new_available_tasks = self.taskmaster.get_available_tasks()
            newly_available = [t for t in new_available_tasks if t not in available_tasks]
            
            # Format the response
            response = f"## Task #{task_id} Implementation\n\n"
            response += f"**{task.title}**\n\n"
            response += result
            
            # Add information about newly available tasks
            if newly_available:
                response += f"\n\n## New Tasks Available\n\n"
                response += f"The following tasks are now available for implementation:\n\n"
                response += self.taskmaster.format_tasks_for_agents(newly_available, show_details=False)
                response += f"\n\nUse 'show task details for #ID' to see implementation details of a specific task.\n"
            
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _implement_task: {e}\n{error_trace}")
            return f"## Error Implementing Task\n\nAn error occurred while implementing the task: {str(e)}\n\nPlease try again or contact support."
    
    async def _list_tasks(self):
        """List all tasks."""
        try:
            self._initialize_taskmaster()
            
            if not self.current_tasks:
                return "No tasks available. Please process an issue first."
            
            # Get available tasks (no dependencies)
            available_tasks = self.taskmaster.get_available_tasks()
            
            # Get completed tasks
            completed_tasks = [t for t in self.current_tasks if t.status == "done"]
            
            response = f"## All Tasks\n\n"
            for task in self.current_tasks:
                response += f"**Task #{task.id}: {task.title}**\n"
                response += f"Description: {task.description}\n"
                response += f"Status: {task.status}\n\n"
            
            # Add section for available tasks
            response += f"\n\n## Ready to Implement Tasks\n"
            if available_tasks:
                response += f"These tasks have no unmet dependencies and can be implemented immediately:\n\n"
                response += self.taskmaster.format_tasks_for_agents(available_tasks, show_details=False)
                
                # Add quick links to details and implementation
                response += f"\n\nCommands:\n"
                response += f"- 'show task details for #ID' to see implementation details of a specific task\n"
                response += f"- 'implement task #ID' to implement a specific task\n"
            else:
                response += f"No tasks are currently ready for implementation.\n"
            
            # Add section for completed tasks if any
            if completed_tasks:
                response += f"\n\n## Completed Tasks\n"
                response += f"These tasks have been completed:\n\n"
                for task in completed_tasks:
                    response += f"- Task #{task.id}: {task.title}\n"
                
                response += f"\n\nYou can create a PR for these completed tasks by typing: 'create PR for completed tasks'\n"
            
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _list_tasks: {e}\n{error_trace}")
            return f"## Error Listing Tasks\n\nAn error occurred while listing tasks: {str(e)}\n\nPlease try again or contact support."
    
    async def _create_pr(self):
        """Create a PR from completed tasks."""
        try:
            self._initialize_taskmaster()
            
            if not self.current_tasks:
                return "No tasks available. Please process an issue first."
            
            # Get completed tasks
            completed_tasks = [t for t in self.current_tasks if t.status == "done"]
            
            if not completed_tasks:
                return "No completed tasks found. Please implement at least one task before creating a PR."
            
            # Validate local repository
            if not self._validate_local_repo():
                return "No valid local repository found. Please set the LOCAL_REPO_PATH environment variable to a valid git repository."
            
            # Create system prompt with PR details
            system_prompt = f"""
            Create a Pull Request for the following completed tasks:
            
            Repository: {self.current_repo_url}
            Local Repository Path: {self.local_repo_path}
            Feature Branch: {self.feature_branch}
            Issue URL: {self.current_issue_url}
            
            Completed Tasks:
            {', '.join([f'#{t.id}: {t.title}' for t in completed_tasks])}
            
            PRD Context:
            {self.current_prd[:500]}...
            
            IMPORTANT: All tasks have been implemented in the same feature branch ({self.feature_branch}).
            """
            
            # Initialize git manager agent
            model_id = self.config_manager.lm_provider_params["model_id"]
            git_manager = Agent(
                name="git_pr_manager",
                role="Git PR Manager",
                model=AwsBedrock(id=model_id, session=session),
                markdown=True,
                instructions=[
                    "PR CREATION WORKFLOW:",
                    
                    "STEP 1 - Repository Verification:",
                    f"   - Verify the local repository at {self.local_repo_path}",
                    f"   - Check that you're on the feature branch {self.feature_branch}",
                    "   - Ensure all changes are committed and pushed",
                    
                    "STEP 2 - PR Description Creation:",
                    "   - Create a detailed PR description based on completed tasks",
                    "   - Include task IDs and titles",
                    "   - Summarize the changes made",
                    "   - Reference the original issue",
                    
                    "STEP 3 - PR Creation Instructions:",
                    "   - Provide clear instructions for the user to create the PR",
                    "   - Include the branch name to use",
                    "   - Include the PR description to copy-paste",
                    
                    "CRITICAL REQUIREMENTS:",
                    "- DO NOT create the PR automatically",
                    "- Provide instructions for the user to create it manually",
                    "- Ensure all task implementations are included"
                ],
                tools=[
                    ShellTools(),
                    GithubTools(get_pull_requests=True),
                    ReasoningTools(add_instructions=True, think=True, analyze=True)
                ]
            )
            
            # Run the git manager to prepare PR
            response = git_manager.run(
                f"Create PR instructions for completed tasks: {', '.join([f'#{t.id}' for t in completed_tasks])}",
                stream=False
            )
            
            return response.content
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _create_pr: {e}\n{error_trace}")
            return f"## Error Creating PR\n\nAn error occurred while creating the PR: {str(e)}\n\nPlease try again or create the PR manually."
    
    async def _standard_pr_creation(self, message_body, system_prompt):
        """Standard PR creation workflow."""
        # Auto-analyze repository if URL is provided
        self._auto_analyze_repo(message_body)
        
        # Add local repository path to system prompt if available
        if self.local_repo_path:
            system_prompt += f"\n\nUse the existing local repository at: {self.local_repo_path}"
        
        team = self.initialize_team(system_prompt)
        response = team.run(
            message_body, 
            stream=False,
            stream_intermediate_steps=True,
            show_full_reasoning=True
        )

        return response.content
    
    async def process_message(self, message: Message):
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

        variables = PRCreationPersonaVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
            context=history_text
        )
        
        system_prompt = PR_CREATION_PROMPT_TEMPLATE.format_messages(**variables.model_dump())[0].content
        
        try:
            # Parse command from message
            command = self._parse_command(message.body)
            action = command["action"]
            
            # Execute appropriate action
            if action == "process_issue":
                response = await self._process_issue(command["issue_url"])
            elif action == "show_task_details":
                response = await self._show_task_details(command["task_id"])
            elif action == "implement_task":
                response = await self._implement_task(command["task_id"])
            elif action == "list_tasks":
                response = await self._list_tasks()
            elif action == "create_pr":
                response = await self._create_pr()
            else:  # standard_pr_creation
                response = await self._standard_pr_creation(message.body, system_prompt)
            
            # Stream response
            async def response_iterator():
                yield response
            
            await self.stream_message(response_iterator())
            
        except ValueError as e:
            error_message = f"Configuration Error: {str(e)}\nThis may be due to missing or invalid environment variables, model configuration, or input parameters."
            async def error_iterator():
                yield error_message
            await self.stream_message(error_iterator())
            
        except boto3.exceptions.Boto3Error as e:
            error_message = f"AWS Connection Error: {str(e)}\nThis may be due to invalid AWS credentials or network connectivity issues."
            async def error_iterator():
                yield error_message
            await self.stream_message(error_iterator())
            
        except Exception as e:
            error_message = f"PR Creation Error ({type(e).__name__}): {str(e)}\nAn unexpected error occurred while the PR creation team was processing your request."
            async def error_iterator():
                yield error_message
            await self.stream_message(error_iterator())
    
    def _validate_local_repo(self):
        """Validate that the local repository path exists and is a git repository.
        If the path doesn't exist but is specified, clone the repository there."""
        if not self.local_repo_path:
            return False
            
        try:
            # Ensure the directory exists
            if not os.path.isdir(self.local_repo_path):
                print(f"Creating directory {self.local_repo_path}")
                os.makedirs(self.local_repo_path, exist_ok=True)
            
            # Check if it's a git repository
            is_git_repo = False
            try:
                result = subprocess.run(
                    ["git", "-C", self.local_repo_path, "rev-parse", "--is-inside-work-tree"],
                    capture_output=True, text=True
                )
                is_git_repo = result.returncode == 0 and result.stdout.strip() == "true"
            except Exception:
                is_git_repo = False
            
            # If not a git repo and we have a repo URL, clone it
            if not is_git_repo and self.current_repo_url:
                print(f"Cloning repository {self.current_repo_url} to {self.local_repo_path}")
                
                # Remove any existing content
                if os.path.exists(self.local_repo_path) and os.listdir(self.local_repo_path):
                    print("Removing existing content before cloning")
                    for item in os.listdir(self.local_repo_path):
                        item_path = os.path.join(self.local_repo_path, item)
                        if os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
                
                # Hardcode the GitHub username
                username = "bhavana-nair"
                print(f"Using hardcoded GitHub username: {username}")
                
                # If we have a username, use the fork URL instead of the original repo URL
                if username and 'github.com' in self.current_repo_url:
                    # Extract original repo owner and name
                    repo_parts = self.current_repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
                    if len(repo_parts) >= 2:
                        original_owner, repo_name = repo_parts[0], repo_parts[1]
                        # Create fork URL
                        fork_url = f"https://github.com/{username}/{repo_name}.git"
                        print(f"Using fork URL: {fork_url} instead of original: {self.current_repo_url}")
                        
                        # Clone from the fork
                        result = subprocess.run(
                            ["git", "clone", fork_url, self.local_repo_path],
                            capture_output=True, text=True
                        )
                        
                        # Add original repo as upstream remote
                        if result.returncode == 0:
                            print("Adding original repository as upstream remote")
                            subprocess.run(
                                ["git", "-C", self.local_repo_path, "remote", "add", "upstream", self.current_repo_url],
                                capture_output=True, text=True
                            )
                    else:
                        print(f"Could not parse repository URL: {self.current_repo_url}, using original URL")
                        result = subprocess.run(
                            ["git", "clone", self.current_repo_url, self.local_repo_path],
                            capture_output=True, text=True
                        )
                else:
                    # Fall back to original URL if we can't determine the fork
                    print(f"Using original repository URL: {self.current_repo_url}")
                    result = subprocess.run(
                        ["git", "clone", self.current_repo_url, self.local_repo_path],
                        capture_output=True, text=True
                    )
                
                if result.returncode != 0:
                    print(f"Failed to clone repository: {result.stderr}")
                    return False
                    
                print(f"Successfully cloned repository to {self.local_repo_path}")
                is_git_repo = True
            elif not is_git_repo:
                print(f"Local repository path {self.local_repo_path} is not a git repository and no repo URL is available")
                return False
            
            # If we have a feature branch, check it out or create it
            if is_git_repo and self.feature_branch:
                print(f"Setting up feature branch: {self.feature_branch}")
                
                # Check if the branch exists
                branch_result = subprocess.run(
                    ["git", "-C", self.local_repo_path, "branch", "--list", self.feature_branch],
                    capture_output=True, text=True
                )
                branch_exists = self.feature_branch in branch_result.stdout if branch_result.stdout else False
                
                if branch_exists:
                    print(f"Checking out existing branch: {self.feature_branch}")
                    subprocess.run(
                        ["git", "-C", self.local_repo_path, "checkout", self.feature_branch],
                        capture_output=True, text=True
                    )
                else:
                    print(f"Creating new branch: {self.feature_branch}")
                    # Try to checkout main or master first
                    try:
                        subprocess.run(
                            ["git", "-C", self.local_repo_path, "checkout", "main"],
                            capture_output=True, text=True
                        )
                    except:
                        try:
                            subprocess.run(
                                ["git", "-C", self.local_repo_path, "checkout", "master"],
                                capture_output=True, text=True
                            )
                        except:
                            print("Could not find main or master branch")
                    
                    # Create and checkout the feature branch
                    try:
                        subprocess.run(
                            ["git", "-C", self.local_repo_path, "checkout", "-b", self.feature_branch],
                            capture_output=True, text=True
                        )
                        print(f"Created and checked out branch: {self.feature_branch}")
                        
                        # Push the new branch to the fork
                        print(f"Pushing new branch to fork: {self.feature_branch}")
                        try:
                            # Check if we're using a fork by looking for upstream remote
                            remotes = subprocess.run(
                                ["git", "-C", self.local_repo_path, "remote", "-v"],
                                capture_output=True, text=True
                            )
                            using_fork = "upstream" in remotes.stdout
                            
                            if using_fork:
                                print(f"Pushing branch to fork (origin/{self.feature_branch})")
                            else:
                                print(f"Pushing branch to origin/{self.feature_branch}")
                                
                            subprocess.run(
                                ["git", "-C", self.local_repo_path, "push", "-u", "origin", self.feature_branch],
                                capture_output=True, text=True
                            )
                            print(f"Successfully pushed branch {self.feature_branch} to fork")
                        except Exception as push_error:
                            print(f"Warning: Could not push branch to fork: {push_error}")
                    except Exception as e:
                        print(f"Warning: Could not create feature branch: {e}")
            
            print(f"Validated local repository at {self.local_repo_path}")
            return True
        except Exception as e:
            print(f"Error validating local repository: {e}")
            return False
    
    def _auto_analyze_repo(self, issue_text: str):
        """Automatically extract repo URL and create knowledge graph"""
        # If we don't have a repo URL yet, try to extract it from the issue text
        if not self.current_repo_url:
            patterns = [
                r'https://github\.com/([^/\s]+/[^/\s]+)',
                r'github\.com/([^/\s]+/[^/\s]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, issue_text)
                if match:
                    repo_path = match.group(1).rstrip('/')
                    self.current_repo_url = f"https://github.com/{repo_path}.git"
                    print(f"Extracted repo URL: {self.current_repo_url}")
                    break
        
        # If we have a valid local repository, use that instead of cloning
        if self._validate_local_repo():
            print(f"Using local repository at {self.local_repo_path} for analysis")
            analyzer = BulkCodeAnalyzer("neo4j://127.0.0.1:7687", (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")))
            analyzer.analyze_folder(self.local_repo_path, clear_existing=True)
            return self.local_repo_path
        
        # If we have a repo URL but no valid local repo, clone it to a temporary location
        if self.current_repo_url:
            return self._clone_and_analyze(self.current_repo_url)
            
        return None
    
    def _clone_and_analyze(self, repo_url: str):
        """Clone repository and create knowledge graph"""
        import time
        start_time = time.time()
        
        try:
            # Use current directory for cloning to avoid path issues
            current_dir = os.getcwd()
            target_folder = os.path.join(current_dir, "repo_analysis")
            
            # Remove existing folder if it exists
            if os.path.exists(target_folder):
                subprocess.run(["rm", "-rf", target_folder], check=True, capture_output=True)
            
            clone_start = time.time()
            
            # Hardcode the GitHub username
            username = "bhavana-nair"
            print(f"Using hardcoded GitHub username: {username}")
            
            # If we have a username, use the fork URL instead of the original repo URL
            if username and 'github.com' in repo_url:
                # Extract original repo owner and name
                repo_parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
                if len(repo_parts) >= 2:
                    original_owner, repo_name = repo_parts[0], repo_parts[1]
                    # Create fork URL
                    fork_url = f"https://github.com/{username}/{repo_name}.git"
                    print(f"Using fork URL: {fork_url} instead of original: {repo_url}")
                    
                    # Clone from the fork
                    subprocess.run(["git", "clone", fork_url, target_folder], check=True, capture_output=True)
                    
                    # Add original repo as upstream remote
                    print("Adding original repository as upstream remote")
                    subprocess.run(
                        ["git", "-C", target_folder, "remote", "add", "upstream", repo_url],
                        capture_output=True, text=True
                    )
                else:
                    print(f"Could not parse repository URL: {repo_url}, using original URL")
                    subprocess.run(["git", "clone", repo_url, target_folder], check=True, capture_output=True)
            else:
                # Fall back to original URL if we can't determine the fork
                print(f"Using original repository URL: {repo_url}")
                subprocess.run(["git", "clone", repo_url, target_folder], check=True, capture_output=True)
                
            clone_time = time.time() - clone_start
            
            kg_start = time.time()
            # Get Neo4j credentials from environment variables
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "")
            if not neo4j_password:
                print("Warning: NEO4J_PASSWORD environment variable not set. Knowledge graph analysis may fail.")
            analyzer = BulkCodeAnalyzer("neo4j://127.0.0.1:7687", (neo4j_user, neo4j_password))
            analyzer.analyze_folder(target_folder, clear_existing=True)
            kg_time = time.time() - kg_start
            
            total_time = time.time() - start_time
            print(f"KG Creation Times - Clone: {clone_time:.2f}s, Analysis: {kg_time:.2f}s, Total: {total_time:.2f}s")
            
            return target_folder
            
        except Exception as e:
            print(f"Error analyzing repository {repo_url}: {e}")
            return None