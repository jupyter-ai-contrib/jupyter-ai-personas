"""Task execution agent that picks up and executes tasks from TaskMaster."""

import os
import subprocess
import sys
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.shell import ShellTools
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from .taskmaster_client import TaskMasterClient, Task

# Add path for repo structure tools
sys.path.append('../pr_creation_persona')
try:
    from jupyter_ai_personas.pr_creation_persona.repo_structure_tools import RepoStructureTools
except ImportError:
    # Fallback path for direct imports
    try:
        from ..pr_creation_persona.repo_structure_tools import RepoStructureTools
    except ImportError:
        print("Warning: Could not import RepoStructureTools")
        RepoStructureTools = None


class TaskExecutionAgent:
    """Agent that can pick up and execute specific tasks."""
    
    def __init__(self, model_id: str, session, agent_name: str = "task_executor"):
        self.taskmaster_client = TaskMasterClient()
        self.agent = Agent(
            name=agent_name,
            role="Task Execution Specialist",
            model=AwsBedrock(id=model_id, session=session),
            markdown=True,
            instructions=[
                "TASK EXECUTION WORKFLOW:",
                
                "STEP 1 - Task Selection:",
                "   - Review available tasks from TaskMaster",
                "   - Select tasks with no unmet dependencies",
                "   - Prioritize high-priority tasks",
                
                "STEP 2 - Task Analysis:",
                "   - Understand task requirements and acceptance criteria",
                "   - Identify required files and components",
                "   - Plan implementation approach",
                
                "STEP 3 - Implementation:",
                "   - Write minimal code to meet acceptance criteria",
                "   - Follow existing code patterns",
                "   - Implement proper error handling",
                
                "STEP 4 - Validation:",
                "   - Verify implementation meets acceptance criteria",
                "   - Test functionality where possible",
                "   - Document any assumptions or limitations",
                
                "EXECUTION PRINCIPLES:",
                "- Focus only on assigned task scope",
                "- Write minimal, clean code",
                "- Follow existing patterns and conventions",
                "- Complete acceptance criteria fully"
            ],
            tools=[
                ShellTools(),
                FileTools(),
                PythonTools(),
                RepoStructureTools() if RepoStructureTools else None
            ]
        )
    
    async def execute_task(self, task: Task, repo_context: str = "") -> str:
        """Execute a specific task."""
        # Extract information from repo_context
        local_repo_path = None
        repo_url = None
        feature_branch = None
        
        for line in repo_context.split('\n'):
            if line.startswith('Local Repository Path:'):
                local_repo_path = line.replace('Local Repository Path:', '').strip()
            elif line.startswith('Repository URL:'):
                repo_url = line.replace('Repository URL:', '').strip()
            elif line.startswith('Feature Branch:'):
                feature_branch = line.replace('Feature Branch:', '').strip()
        
        # Set up git repository if needed
        if local_repo_path and repo_url:
            self._setup_git_repository(local_repo_path, repo_url, feature_branch)
                
        prompt = f"""
        Execute the following task:
        
        TASK: {task.title}
        ID: {task.id}
        PRIORITY: {task.priority}
        
        DESCRIPTION:
        {task.description}
        
        DETAILS:
        {task.details or 'No additional details provided'}
        
        TEST STRATEGY:
        {task.test_strategy or 'No test strategy specified'}
        
        DEPENDENCIES: {', '.join(task.dependencies) if task.dependencies else 'None'}
        
        REPOSITORY CONTEXT:
        {repo_context}
        
        Implement this task following the execution workflow.
        Focus only on meeting the acceptance criteria with minimal code.
        
        IMPORTANT INSTRUCTIONS:
        - Save all files to: {local_repo_path}
        - NEVER create files in the repository root
        - Follow project structure conventions and place files in appropriate directories
        - Create parent directories if they don't exist before creating files
        - Use the feature branch: {feature_branch or 'main'}
        - After implementing the code, commit your changes with a descriptive message
        - Include the task ID in your commit message
        """
        
        # If local_repo_path is available, set the working directory for all tools
        if local_repo_path:
            print(f"Setting working directory to: {local_repo_path}")
            # Set ShellTools working directory
            self.agent.tools[0].cwd = local_repo_path
            # Set FileTools base path
            self.agent.tools[1].base_path = local_repo_path
            
            # Analyze repository structure if RepoStructureTools is available
            structure_info = ""
            if RepoStructureTools and os.path.exists(local_repo_path):
                try:
                    print("\nAnalyzing repository structure before task execution...")
                    structure_tools = RepoStructureTools()
                    
                    # Analyze folder structure
                    structure_analysis = structure_tools.analyze_folder_structure(None, local_repo_path)
                    
                    # Get component placement map
                    placement_map = structure_tools.get_component_placement_map(None, local_repo_path)
                    
                    # Analyze project templates
                    template_analysis = structure_tools.analyze_project_templates(None, local_repo_path)
                    
                    # Add structure information to prompt
                    structure_info = "\n\nREPOSITORY STRUCTURE ANALYSIS:\n"
                    structure_info += "Component Placement Guidelines:\n"
                    
                    # Extract component placement guidelines from placement map
                    for line in placement_map.split('\n'):
                        if line.startswith('- ') and ': ' in line:
                            structure_info += f"{line}\n"
                    
                    # Add template information
                    structure_info += "\nProject Templates:\n"
                    template_lines = template_analysis.split('\n')
                    for line in template_lines[:10]:  # Limit to first 10 lines
                        structure_info += f"{line}\n"
                    
                    print("Repository structure analysis complete")
                except Exception as e:
                    print(f"Warning: Error analyzing repository structure: {e}")
            
            # Add explicit instructions about the path
            prompt += f"""
            
            CRITICAL PATH INSTRUCTIONS:
            - You MUST save all files to: {local_repo_path}
            - Use absolute paths when creating files
            - NEVER create files directly in the repository root
            - Follow the project structure patterns identified below
            - Create parent directories if they don't exist before creating files
            - Validate file paths against project conventions
            {structure_info}
            """
        else:
            print("Warning: No local repository path specified. Files will be saved to the current directory.")
            
        # Execute the task
        response = self.agent.run(prompt, stream=False)
        
        # If we have a local repo and feature branch, commit the changes
        if local_repo_path and feature_branch:
            try:
                # Check if there are changes to commit
                status_result = subprocess.run(
                    ["git", "-C", local_repo_path, "status", "--porcelain"],
                    capture_output=True, text=True
                )
                
                if status_result.stdout.strip():
                    print("Committing changes...")
                    # Add all changes
                    subprocess.run(
                        ["git", "-C", local_repo_path, "add", "."],
                        check=True, capture_output=True
                    )
                    
                    # Commit changes
                    commit_message = f"Implement task #{task.id}: {task.title}"
                    subprocess.run(
                        ["git", "-C", local_repo_path, "commit", "-m", commit_message],
                        check=True, capture_output=True
                    )
                    
                    print(f"Changes committed to branch {feature_branch}")
                    
                    # Try to push changes to the fork (origin)
                    try:
                        # Check if we're using a fork by looking for upstream remote
                        remotes = subprocess.run(
                            ["git", "-C", local_repo_path, "remote", "-v"],
                            capture_output=True, text=True
                        )
                        using_fork = "upstream" in remotes.stdout
                        
                        if using_fork:
                            print(f"Pushing changes to fork (origin/{feature_branch})")
                        else:
                            print(f"Pushing changes to origin/{feature_branch}")
                        
                        # First try to get more detailed error information
                        print("Checking git status before push...")
                        subprocess.run(
                            ["git", "-C", local_repo_path, "status"],
                            capture_output=False
                        )
                        
                        # Try pushing with verbose output
                        push_result = subprocess.run(
                            ["git", "-C", local_repo_path, "push", "-v", "-u", "origin", feature_branch],
                            capture_output=True, text=True
                        )
                        
                        if push_result.returncode == 0:
                            print(f"Changes pushed to remote branch {feature_branch}")
                        else:
                            print(f"Push failed with error code {push_result.returncode}")
                            print(f"Error output: {push_result.stderr}")
                            print(f"Standard output: {push_result.stdout}")
                            
                            # Try to diagnose the issue
                            print("\nAttempting to diagnose push failure...")
                            print("\nChecking remote URLs:")
                            subprocess.run(
                                ["git", "-C", local_repo_path, "remote", "-v"],
                                capture_output=False
                            )
                            
                            # Continue with task completion even if push fails
                            print("\nContinuing with task completion despite push failure")
                    except Exception as e:
                        print(f"Warning: Could not push changes: {e}")
                        print("\nContinuing with task completion despite push failure")
                else:
                    print("No changes to commit")
            except Exception as e:
                print(f"Warning: Error during git operations: {e}")
        
        return response.content if hasattr(response, 'content') else str(response)
            
    def _setup_git_repository(self, local_repo_path, repo_url, feature_branch):
        """Set up git repository for task implementation."""
        try:
            print(f"Setting up git repository at {local_repo_path}")
            print(f"Repository URL: {repo_url}")
            print(f"Feature branch: {feature_branch}")
            
            # Step 1: Ensure the directory exists
            os.makedirs(local_repo_path, exist_ok=True)
            
            # Step 2: Check if it's already a git repository
            is_git_repo = os.path.exists(os.path.join(local_repo_path, '.git'))
            
            # Step 3: Clone the repository if needed
            if not is_git_repo:
                print(f"Cloning repository {repo_url} to {local_repo_path}")
                # Remove any existing content
                if os.path.exists(local_repo_path) and os.listdir(local_repo_path):
                    print("Removing existing content before cloning")
                    for item in os.listdir(local_repo_path):
                        item_path = os.path.join(local_repo_path, item)
                        if os.path.isdir(item_path) and item != '.git':
                            import shutil
                            shutil.rmtree(item_path)
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
                
                # Clone the repository from the fork instead of the original repo
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
                        result = subprocess.run(
                            ["git", "clone", fork_url, local_repo_path],
                            capture_output=True,
                            text=True
                        )
                        
                        # Add original repo as upstream remote
                        if result.returncode == 0:
                            print("Adding original repository as upstream remote")
                            subprocess.run(
                                ["git", "-C", local_repo_path, "remote", "add", "upstream", repo_url],
                                capture_output=True,
                                text=True
                            )
                    else:
                        print(f"Could not parse repository URL: {repo_url}, using original URL")
                        result = subprocess.run(
                            ["git", "clone", repo_url, local_repo_path],
                            capture_output=True,
                            text=True
                        )
                else:
                    # Fall back to original URL if we can't determine the fork
                    print(f"Using original repository URL: {repo_url}")
                    result = subprocess.run(
                        ["git", "clone", repo_url, local_repo_path],
                        capture_output=True,
                        text=True
                    )
                if result.returncode == 0:
                    print("Repository cloned successfully")
                else:
                    print(f"Error cloning repository: {result.stderr}")
                    return False
            else:
                print(f"Using existing git repository at {local_repo_path}")
                # Fetch latest changes
                result = subprocess.run(
                    ["git", "-C", local_repo_path, "fetch"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("Fetched latest changes")
                else:
                    print(f"Warning: Could not fetch latest changes: {result.stderr}")
                    # Continue anyway
            
            # Step 4: Create and checkout the feature branch
            if feature_branch:
                # First checkout main branch
                print("Checking out main branch")
                result = subprocess.run(
                    ["git", "-C", local_repo_path, "checkout", "main"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # Try master if main doesn't exist
                    print("Main branch not found, trying master")
                    result = subprocess.run(
                        ["git", "-C", local_repo_path, "checkout", "master"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        print("Could not find main or master branch, using current branch")
                
                # Create and checkout the feature branch
                print(f"Creating and checking out feature branch: {feature_branch}")
                try:
                    # Check if branch exists
                    branch_exists = subprocess.run(
                        ["git", "-C", local_repo_path, "rev-parse", "--verify", feature_branch],
                        capture_output=True
                    ).returncode == 0
                    
                    if branch_exists:
                        print(f"Branch {feature_branch} already exists, checking it out")
                        subprocess.run(
                            ["git", "-C", local_repo_path, "checkout", feature_branch],
                            check=True
                        )
                    else:
                        print(f"Creating new branch: {feature_branch}")
                        subprocess.run(
                            ["git", "-C", local_repo_path, "checkout", "-b", feature_branch],
                            check=True
                        )
                        
                        # Push the new branch to the fork
                        print(f"Pushing new branch to fork: {feature_branch}")
                        try:
                            # Check remote configuration
                            print("Checking remote configuration:")
                            subprocess.run(
                                ["git", "-C", local_repo_path, "remote", "-v"],
                                capture_output=False
                            )
                            
                            # Try pushing with verbose output
                            push_result = subprocess.run(
                                ["git", "-C", local_repo_path, "push", "-v", "-u", "origin", feature_branch],
                                capture_output=True, text=True
                            )
                            
                            if push_result.returncode == 0:
                                print(f"Successfully pushed branch {feature_branch} to fork")
                            else:
                                print(f"Push failed with error code {push_result.returncode}")
                                print(f"Error output: {push_result.stderr}")
                                print(f"Standard output: {push_result.stdout}")
                                print("\nContinuing with branch setup despite push failure")
                        except Exception as push_error:
                            print(f"Warning: Could not push branch to fork: {push_error}")
                            print("Continuing with branch setup despite push failure")
                except subprocess.CalledProcessError as e:
                    print(f"Error with branch operations: {e}")
            
            print("Git repository setup complete")
            return True
        except Exception as e:
            print(f"Error setting up git repository: {e}")
            return False
    
    def get_available_tasks(self) -> list[Task]:
        """Get tasks that this agent can execute."""
        return self.taskmaster_client.get_available_tasks()
    
    def mark_task_complete(self, task_id: str) -> bool:
        """Mark a task as completed in TaskMaster."""
        return self.taskmaster_client.update_task_status(task_id, 'done')
    
    def mark_task_in_progress(self, task_id: str) -> bool:
        """Mark a task as in progress in TaskMaster."""
        return self.taskmaster_client.update_task_status(task_id, 'in-progress')
        
    def validate_file_path(self, file_path: str, repo_path: str) -> tuple[bool, str]:
        """Validate if a file path follows project conventions.
        
        Args:
            file_path: Path to validate
            repo_path: Path to the repository root
            
        Returns:
            tuple: (is_valid, message)
        """
        if not RepoStructureTools:
            # If RepoStructureTools is not available, consider all paths valid
            return True, "Path validation skipped: RepoStructureTools not available"
            
        try:
            # Create RepoStructureTools instance
            structure_tools = RepoStructureTools()
            
            # Validate the path
            validation_result = structure_tools.validate_file_path(None, file_path, repo_path)
            
            # Check if path is directly in root
            if validation_result.startswith("WARNING:"):
                return False, validation_result
            elif validation_result.startswith("VALID:"):
                return True, validation_result
            else:
                # Path doesn't match known patterns, but might still be valid
                return True, validation_result
                
        except Exception as e:
            # If validation fails, consider the path valid to avoid blocking
            return True, f"Path validation error: {str(e)}"
            
    def ensure_parent_directories(self, file_path: str) -> bool:
        """Ensure parent directories exist before file creation.
        
        Args:
            file_path: Path to the file to be created
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get parent directory
            parent_dir = os.path.dirname(file_path)
            
            # Check if parent directory exists
            if not os.path.exists(parent_dir):
                # Create parent directories
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Created parent directories for: {file_path}")
                
            return True
            
        except Exception as e:
            print(f"Error creating parent directories: {str(e)}")
            return False