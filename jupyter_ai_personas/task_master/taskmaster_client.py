"""TaskMaster AI client integration using the actual TaskMaster library."""

import os
import json
import tempfile
import subprocess
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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


@dataclass
class Task:
    """Represents a task from TaskMaster."""
    id: str
    title: str
    description: str
    priority: str
    status: str
    dependencies: List[str]
    details: Optional[str] = None
    test_strategy: Optional[str] = None


class TaskMasterClient:
    """Client for integrating with actual TaskMaster AI."""
    
    def __init__(self, project_root: str = None):
        # Use a persistent directory for TaskMaster
        if project_root:
            self.project_root = project_root
        else:
            home_dir = os.path.expanduser("~")
            self.project_root = os.path.join(home_dir, ".jupyter-ai-taskmaster")
            os.makedirs(self.project_root, exist_ok=True)
            
        self.tasks: List[Task] = []
        self._taskmaster_available = True
        self._ensure_taskmaster_setup()
    
    def _ensure_taskmaster_setup(self):
        """Check if TaskMaster is available and set up config."""
        try:
            # Check if npx is available
            print("Checking npx availability...")
            subprocess.run(['npx', '--version'], capture_output=True, check=True)
            print("npx is available")
            
            # Check if task-master is available
            print("Checking task-master availability...")
            version_result = subprocess.run(['npx', 'task-master', '--version'], 
                                         capture_output=True, text=True)
            print(f"task-master version: {version_result.stdout.strip() if version_result.returncode == 0 else 'not available'}")
            if version_result.returncode != 0:
                print(f"task-master check failed: {version_result.stderr}")
                self._taskmaster_available = False
                return
            
            # Create config directory with API key
            work_dir = os.getcwd()
            config_dir = os.path.join(work_dir, ".taskmaster")
            os.makedirs(config_dir, exist_ok=True)
            
            # Get API key from environment variable
            api_key = os.getenv('ANTHROPIC_API_KEY')
            print(f"ANTHROPIC_API_KEY: {api_key}")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
            # Create config file with Anthropic API
            config_path = os.path.join(config_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "apiKey": api_key
                }, f, indent=2)
            print("Using Anthropic API for TaskMaster")
                
            print(f"Created TaskMaster config at {config_path}")
            self._taskmaster_available = True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: TaskMaster not available or timed out: {e}")
            self._taskmaster_available = False
        except ValueError as e:
            print(f"Configuration error: {e}")
            self._taskmaster_available = False
    
    async def create_tasks_from_prd(self, prd_content: str) -> List[Task]:
        """Create tasks from PRD using TaskMaster with Anthropic API."""
            
        # Check for API key in environment
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("ANTHROPIC_API_KEY not set, using fallback task creation")
            return
            
        try:
            # Use current directory where TaskMaster is already installed
            work_dir = os.getcwd()
            print(f"Using current directory for TaskMaster: {work_dir}")
            
            # Create PRD file
            prd_path = os.path.join(work_dir, 'prd.txt')
            with open(prd_path, 'w') as f:
                f.write(prd_content)
            
            # Use TaskMaster to parse PRD and generate tasks
            print(f"Running TaskMaster parse-prd on {prd_path}...")
            print(f"Command: npx task-master parse-prd --input {prd_path} --force")
            print(f"Working directory: {work_dir}")
            try:
                # Run parse-prd in the current directory
                result = subprocess.run([
                    'npx', 'task-master', 'parse-prd', 
                    '--input', prd_path,
                    '--force'
                ], cwd=work_dir, capture_output=True, text=True)
                print(f"Command completed with return code: {result.returncode}")
            except Exception as cmd_error:
                print(f"Command execution error: {cmd_error}")
                return 
            
            print(f"TaskMaster parse-prd result: {result.returncode}")
            if result.returncode != 0:
                print(f"TaskMaster stderr: {result.stderr}")
                print(f"TaskMaster stdout: {result.stdout}")
                # Try running with --debug flag to get more information
                debug_result = subprocess.run([
                    'npx', 'task-master', 'parse-prd', 
                    '--input', prd_path,
                    '--force',
                    '--debug'
                ], cwd=work_dir, capture_output=True, text=True)
                print(f"Debug output: {debug_result.stdout}")
                print(f"Debug errors: {debug_result.stderr}")
            
            try:
                return self._load_tasks()
            except Exception as load_error:
                print(f"Error loading tasks: {load_error}")
                
        except Exception as e:
            print(f"Error creating tasks: {e}")
            return 
    
    def _load_tasks(self) -> List[Task]:
        """Load tasks from TaskMaster tasks.json file."""
        # Look for tasks.json in various possible locations
        work_dir = os.getcwd()
        possible_paths = [
            os.path.join(work_dir, "tasks.json"),
            os.path.join(work_dir, ".taskmaster", "tasks", "tasks.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                tasks_path = path
                print(f"Found tasks at: {tasks_path}")
                break
        else:
            print("Tasks file not found")
            return []
        
        if not os.path.exists(tasks_path):
            return []
        
        try:
            with open(tasks_path, 'r') as f:
                data = json.load(f)
            
            tasks = []
            # Check if tasks are nested under 'master' key
            if 'master' in data and 'tasks' in data['master']:
                task_list = data['master']['tasks']
            else:
                task_list = data.get('tasks', [])
                
            for task_data in task_list:
                task = Task(
                    id=str(task_data.get('id', '')),  # Convert ID to string
                    title=task_data.get('title', ''),
                    description=task_data.get('description', ''),
                    priority=task_data.get('priority', 'medium'),
                    status=task_data.get('status', 'pending'),
                    dependencies=[str(dep) for dep in task_data.get('dependencies', [])],  # Convert dependencies to strings
                    details=task_data.get('details'),
                    test_strategy=task_data.get('testStrategy')
                )
                tasks.append(task)
            
            self.tasks = tasks
            return tasks
            
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return []
    
    def get_available_tasks(self) -> List[Task]:
        """Get tasks that can be worked on (no unmet dependencies)."""
        completed_tasks = {t.id for t in self.tasks if t.status == 'done'}
        
        available = []
        for task in self.tasks:
            if task.status == 'pending':
                if not task.dependencies or all(dep in completed_tasks for dep in task.dependencies):
                    available.append(task)
        
        return available
    
    def format_tasks_for_agents(self, tasks: List[Task], show_details: bool = False, repo_path: str = None) -> str:
        """Format tasks for agent consumption.
        
        Args:
            tasks: List of tasks to format
            show_details: Whether to show implementation details and test strategy
            repo_path: Optional repository path for file path suggestions
        """
        if not tasks:
            return "No tasks available."
        
        formatted = ""  # Remove the header to make it cleaner
        for task in tasks:
            formatted += f"**Task #{task.id}: {task.title}**\n"
            formatted += f"Description: {task.description}\n"
            
            # Only show these details if explicitly requested
            if show_details:
                formatted += f"Priority: {task.priority}\n"
                formatted += f"Status: {task.status}\n"
                
                if task.dependencies:
                    formatted += f"Dependencies: {', '.join(task.dependencies)}\n"
                
                # Add file path suggestions if repo_path is provided
                if repo_path and RepoStructureTools and show_details:
                    # Check if task details already include file path suggestions
                    if task.details and not "SUGGESTED FILE PATHS:" in task.details:
                        file_path_suggestions = self._generate_file_path_suggestions(task, repo_path)
                        if file_path_suggestions:
                            # Update task details with file path suggestions
                            if task.details:
                                task.details += "\n\n" + file_path_suggestions
                            else:
                                task.details = file_path_suggestions
                
                if task.details:
                    formatted += f"Implementation Details:\n{task.details}\n"
                
                if task.test_strategy:
                    formatted += f"Test Strategy:\n{task.test_strategy}\n"
            
            formatted += "\n"
        
        return formatted
        
    def _generate_file_path_suggestions(self, task: Task, repo_path: str) -> str:
        """Generate file path suggestions for a task based on repository structure.
        
        Args:
            task: The task to generate suggestions for
            repo_path: Path to the repository root
            
        Returns:
            str: File path suggestions formatted as markdown
        """
        import re
        
        if not RepoStructureTools or not os.path.exists(repo_path):
            return ""
            
        try:
            # Extract potential file names from task description and details
            potential_files = []
            
            # Look for Python class or file mentions in description
            if task.description:
                class_matches = re.findall(r'class\s+([A-Za-z0-9_]+)', task.description)
                file_matches = re.findall(r'([A-Za-z0-9_]+\.py)', task.description)
                potential_files.extend(class_matches)
                potential_files.extend(file_matches)
            
            # Look for Python class or file mentions in details
            if task.details:
                class_matches = re.findall(r'class\s+([A-Za-z0-9_]+)', task.details)
                file_matches = re.findall(r'([A-Za-z0-9_]+\.py)', task.details)
                potential_files.extend(class_matches)
                potential_files.extend(file_matches)
            
            # Convert class names to potential file names if needed
            for i, name in enumerate(potential_files):
                if not name.endswith('.py'):
                    potential_files[i] = f"{name.lower()}.py"
            
            # Remove duplicates
            potential_files = list(set(potential_files))
            
            if not potential_files:
                return ""
                
            # Create RepoStructureTools instance
            structure_tools = RepoStructureTools()
            
            # Generate path suggestions
            file_path_suggestions = "SUGGESTED FILE PATHS:\n"
            
            for file_name in potential_files[:3]:  # Limit to first 3 files
                # Determine component type from filename
                component_type = 'utils'  # Default
                if 'test' in file_name.lower():
                    component_type = 'tests'
                elif 'model' in file_name.lower():
                    component_type = 'models'
                elif 'view' in file_name.lower():
                    component_type = 'views'
                elif 'controller' in file_name.lower():
                    component_type = 'controllers'
                elif 'service' in file_name.lower():
                    component_type = 'services'
                elif 'persona' in file_name.lower():
                    component_type = 'personas'
                
                # Get suggested path
                suggested_path = structure_tools.suggest_file_path(None, file_name, component_type, repo_path)
                file_path_suggestions += f"- {file_name}: {suggested_path}\n"
            
            return file_path_suggestions
            
        except Exception as e:
            print(f"Error generating file path suggestions: {str(e)}")
            return ""
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status using TaskMaster."""
        try:
            # Run TaskMaster command to update status with the correct format
            work_dir = os.getcwd()
            result = subprocess.run([
                'npx', 'task-master', 'set-status',
                f'--status={status}',  # Correct format: --status=done
                f'--id={task_id}'      # Correct format: --id=129
            ], cwd=work_dir, capture_output=True, text=True)
            
            print(f"TaskMaster set-status result: {result.returncode}")
            if result.returncode != 0:
                print(f"Error output: {result.stderr}")
                print(f"Standard output: {result.stdout}")
            else:
                print(f"Successfully updated task {task_id} status to {status}")
            
            # Also update the status in our local tasks list
            for task in self.tasks:
                if task.id == task_id:
                    task.status = status
                    break
            
            # Directly update the tasks.json file as a backup method
            self._update_tasks_json_directly(task_id, status)
            
            # Reload tasks to ensure we have the latest data
            self._load_tasks()
            
            return result.returncode == 0
        except Exception as e:
            print(f"Error updating task status: {e}")
            # Try direct update as fallback
            return self._update_tasks_json_directly(task_id, status)
            
    def _update_tasks_json_directly(self, task_id: str, status: str) -> bool:
        """Directly update the task status in the tasks.json file."""
        try:
            # Find the tasks.json file
            work_dir = os.getcwd()
            tasks_path = os.path.join(work_dir, ".taskmaster", "tasks", "tasks.json")
            
            if not os.path.exists(tasks_path):
                print(f"Tasks file not found at {tasks_path}")
                return False
                
            # Read the current content
            with open(tasks_path, 'r') as f:
                data = json.load(f)
                
            # Update the task status
            updated = False
            if 'master' in data and 'tasks' in data['master']:
                for task in data['master']['tasks']:
                    if str(task.get('id', '')) == str(task_id):
                        task['status'] = status
                        updated = True
                        break
            
            # Write the updated content back
            if updated:
                with open(tasks_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Directly updated task {task_id} status to {status} in {tasks_path}")
                return True
            else:
                print(f"Task {task_id} not found in {tasks_path}")
                return False
                
        except Exception as e:
            print(f"Error directly updating tasks.json: {e}")
            return False
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_task_details(self, task_id: str) -> str:
        """Get formatted details for a specific task."""
        task = self.get_task_by_id(task_id)
        if not task:
            return f"Task with ID {task_id} not found."
        
        # Format with full details
        return self.format_tasks_for_agents([task], show_details=True)
        
    def analyze_repository_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and create folder map for component placement.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dict with structure analysis results
        """
        if not RepoStructureTools or not repo_path or not os.path.exists(repo_path):
            print(f"Cannot analyze repository structure: RepoStructureTools not available or invalid path")
            return {}
            
        try:
            # Create RepoStructureTools instance
            structure_tools = RepoStructureTools()
            
            # Analyze folder structure
            print("Analyzing repository folder structure...")
            structure_analysis = structure_tools.analyze_folder_structure(None, repo_path)
            
            # Get component placement map
            print("\nGenerating component placement map...")
            placement_map = structure_tools.get_component_placement_map(None, repo_path)
            
            # Analyze project templates
            print("\nAnalyzing project templates...")
            template_analysis = structure_tools.analyze_project_templates(None, repo_path)
            
            # Parse the placement map to create folder_map
            folder_map = {}
            for line in placement_map.split('\n'):
                if ': ' in line and line.startswith('- '):
                    comp_type, folder = line.replace('- ', '').split(': ', 1)
                    folder_map[comp_type.strip()] = folder.strip()
                    
            print(f"\nFolder map created: {folder_map}")
            
            return {
                "structure_analysis": structure_analysis,
                "placement_map": placement_map,
                "template_analysis": template_analysis,
                "folder_map": folder_map
            }
            
        except Exception as e:
            print(f"Error analyzing repository structure: {str(e)}")
            return {}
  