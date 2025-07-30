"""Repository structure analysis tools for PR creation persona."""

import os
import re
from typing import Dict, List, Tuple, Optional
from agno.tools import Toolkit
from agno.agent import Agent
import sys
sys.path.append('../knowledge_graph')
from jupyter_ai_personas.knowledge_graph.code_analysis_tool import CodeAnalysisTool

class RepoStructureTools(Toolkit):
    """Tools for analyzing repository structure and determining proper file placement."""
    
    def __init__(self, **kwargs):
        self.code_tool = CodeAnalysisTool()
        
        super().__init__(name="repo_structure", tools=[
            self.analyze_folder_structure,
            self.get_component_placement_map,
            self.suggest_file_path,
            self.validate_file_path,
            self.create_parent_directories,
            self.analyze_project_templates
        ], **kwargs)
    
    def analyze_folder_structure(self, agent: Agent, repo_path: str) -> str:
        """
        Analyze the repository folder structure to identify patterns.
        
        Args:
            agent: The agent instance
            repo_path: Path to the repository root
            
        Returns:
            str: Analysis of folder structure patterns
        """
        try:
            # Get folder structure
            folder_map = {}
            component_types = {
                "models": [],
                "views": [],
                "controllers": [],
                "utils": [],
                "tests": [],
                "services": [],
                "components": [],
                "personas": []
            }
            
            # Walk through the repository
            for root, dirs, files in os.walk(repo_path):
                # Skip common non-code directories
                if any(skip_dir in root for skip_dir in ['.git', 'node_modules', '__pycache__', '.venv']):
                    continue
                
                # Analyze Python files
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                        folder = os.path.dirname(rel_path)
                        
                        # Track folder usage
                        folder_map[folder] = folder_map.get(folder, 0) + 1
                        
                        # Categorize components based on patterns
                        if 'model' in file.lower() or 'schema' in file.lower():
                            component_types["models"].append(rel_path)
                        elif 'view' in file.lower() or 'template' in file.lower():
                            component_types["views"].append(rel_path)
                        elif 'controller' in file.lower() or 'handler' in file.lower():
                            component_types["controllers"].append(rel_path)
                        elif 'util' in file.lower() or 'helper' in file.lower():
                            component_types["utils"].append(rel_path)
                        elif 'test' in file.lower():
                            component_types["tests"].append(rel_path)
                        elif 'service' in file.lower():
                            component_types["services"].append(rel_path)
                        elif 'component' in file.lower():
                            component_types["components"].append(rel_path)
                        elif 'persona' in file.lower():
                            component_types["personas"].append(rel_path)
            
            # Generate analysis
            analysis = "Repository Structure Analysis:\n\n"
            
            # Most common folders
            sorted_folders = sorted(folder_map.items(), key=lambda x: x[1], reverse=True)
            analysis += "Common Code Folders:\n"
            for folder, count in sorted_folders[:10]:
                analysis += f"- {folder}: {count} files\n"
            
            # Component type patterns
            analysis += "\nComponent Type Patterns:\n"
            for comp_type, paths in component_types.items():
                if paths:
                    common_folders = {}
                    for path in paths:
                        folder = os.path.dirname(path)
                        common_folders[folder] = common_folders.get(folder, 0) + 1
                    
                    most_common = sorted(common_folders.items(), key=lambda x: x[1], reverse=True)
                    if most_common:
                        analysis += f"- {comp_type}: typically in {most_common[0][0]}\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing folder structure: {str(e)}"
    
    def get_component_placement_map(self, agent: Agent, repo_path: str) -> str:
        """
        Create a mapping of component types to appropriate folder locations.
        
        Args:
            agent: The agent instance
            repo_path: Path to the repository root
            
        Returns:
            str: JSON mapping of component types to folder locations
        """
        try:
            # Initialize component type mapping
            component_map = {}
            
            # Walk through the repository
            for root, dirs, files in os.walk(repo_path):
                # Skip common non-code directories
                if any(skip_dir in root for skip_dir in ['.git', 'node_modules', '__pycache__', '.venv']):
                    continue
                
                # Check for specific component folders
                rel_path = os.path.relpath(root, repo_path)
                
                # Map component types based on folder names
                folder_name = os.path.basename(root).lower()
                if 'test' in folder_name:
                    component_map['tests'] = rel_path
                elif 'model' in folder_name:
                    component_map['models'] = rel_path
                elif 'view' in folder_name:
                    component_map['views'] = rel_path
                elif 'controller' in folder_name:
                    component_map['controllers'] = rel_path
                elif 'util' in folder_name or 'helper' in folder_name:
                    component_map['utils'] = rel_path
                elif 'service' in folder_name:
                    component_map['services'] = rel_path
                elif 'component' in folder_name:
                    component_map['components'] = rel_path
                elif 'persona' in folder_name:
                    component_map['personas'] = rel_path
                
                # Look for patterns in Python files
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Check for class patterns
                                if re.search(r'class\s+\w+Model|\w+Schema', content):
                                    component_map.setdefault('models', rel_path)
                                if re.search(r'class\s+\w+View|\w+Template', content):
                                    component_map.setdefault('views', rel_path)
                                if re.search(r'class\s+\w+Controller|\w+Handler', content):
                                    component_map.setdefault('controllers', rel_path)
                                if re.search(r'class\s+\w+Service', content):
                                    component_map.setdefault('services', rel_path)
                                if re.search(r'class\s+\w+Persona', content):
                                    component_map.setdefault('personas', rel_path)
                        except:
                            # Skip files that can't be read
                            pass
            
            # Format the result
            result = "Component Placement Map:\n"
            for comp_type, folder in component_map.items():
                result += f"- {comp_type}: {folder}\n"
            
            return result
            
        except Exception as e:
            return f"Error creating component placement map: {str(e)}"
    
    def suggest_file_path(self, agent: Agent, file_name: str, component_type: str, repo_path: str) -> str:
        """
        Suggest appropriate file path based on repository patterns.
        
        Args:
            agent: The agent instance
            file_name: Name of the file to create
            component_type: Type of component (model, view, controller, etc.)
            repo_path: Path to the repository root
            
        Returns:
            str: Suggested file path
        """
        try:
            # Get component placement map
            placement_map_str = self.get_component_placement_map(agent, repo_path)
            
            # Extract folder for component type
            component_folder = None
            for line in placement_map_str.split('\n'):
                if line.startswith(f"- {component_type}:"):
                    component_folder = line.split(': ')[1].strip()
                    break
            
            # If no specific folder found, look for similar components
            if not component_folder:
                # Use knowledge graph to find similar components
                query = f"""
                MATCH (n) 
                WHERE n.name CONTAINS '{file_name.replace('.py', '')}' OR 
                      n.name CONTAINS '{component_type}'
                RETURN n.file as file_path
                LIMIT 5
                """
                
                try:
                    result = self.code_tool.query_code(query)
                    if result and isinstance(result, list) and len(result) > 0:
                        # Extract common folder pattern
                        folders = []
                        for item in result:
                            if 'file_path' in item and item['file_path']:
                                folder = os.path.dirname(item['file_path'])
                                folders.append(folder)
                        
                        if folders:
                            # Find most common folder
                            folder_counts = {}
                            for folder in folders:
                                folder_counts[folder] = folder_counts.get(folder, 0) + 1
                            
                            most_common = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
                            if most_common:
                                component_folder = most_common[0][0]
                except:
                    # If knowledge graph query fails, continue with other methods
                    pass
            
            # If still no folder found, use common patterns
            if not component_folder:
                # Default patterns based on component type
                if component_type == 'tests':
                    component_folder = 'tests'
                elif component_type == 'models':
                    component_folder = 'models'
                elif component_type == 'views':
                    component_folder = 'views'
                elif component_type == 'controllers':
                    component_folder = 'controllers'
                elif component_type == 'utils':
                    component_folder = 'utils'
                elif component_type == 'services':
                    component_folder = 'services'
                elif component_type == 'personas':
                    # Look for persona pattern
                    for root, dirs, files in os.walk(repo_path):
                        if 'persona.py' in files:
                            rel_path = os.path.relpath(root, repo_path)
                            component_folder = rel_path
                            break
                    
                    # If still not found, use default pattern
                    if not component_folder:
                        component_folder = 'jupyter_ai_personas'
            
            # Construct suggested path
            if component_folder:
                suggested_path = os.path.join(repo_path, component_folder, file_name)
            else:
                # Default to a reasonable location if no pattern found
                suggested_path = os.path.join(repo_path, 'jupyter_ai_personas', file_name)
            
            return suggested_path
            
        except Exception as e:
            return f"Error suggesting file path: {str(e)}"
    
    def validate_file_path(self, agent: Agent, file_path: str, repo_path: str) -> str:
        """
        Validate if a file path follows project conventions.
        
        Args:
            agent: The agent instance
            file_path: Path to validate
            repo_path: Path to the repository root
            
        Returns:
            str: Validation result with suggestions if needed
        """
        try:
            # Make path relative to repo root
            if file_path.startswith(repo_path):
                rel_path = os.path.relpath(file_path, repo_path)
            else:
                rel_path = file_path
            
            # Check if path is directly in root (which we want to avoid)
            if '/' not in rel_path and '\\' not in rel_path:
                # File is in root, suggest better location
                file_name = os.path.basename(rel_path)
                
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
                
                # Get better suggestion
                better_path = self.suggest_file_path(agent, file_name, component_type, repo_path)
                
                return f"WARNING: File would be created in repository root. Consider using: {better_path}"
            
            # Check if path follows existing patterns
            placement_map_str = self.get_component_placement_map(agent, repo_path)
            
            # Determine if path matches any known patterns
            matches_pattern = False
            for line in placement_map_str.split('\n'):
                if ': ' in line:
                    _, folder = line.split(': ', 1)
                    if folder.strip() in rel_path:
                        matches_pattern = True
                        break
            
            if matches_pattern:
                return f"VALID: Path follows project conventions: {file_path}"
            else:
                # Path doesn't match known patterns, but might still be valid
                return f"CAUTION: Path doesn't match common project patterns, but may still be valid: {file_path}"
            
        except Exception as e:
            return f"Error validating file path: {str(e)}"
    
    def create_parent_directories(self, agent: Agent, file_path: str) -> str:
        """
        Ensure parent directories exist before file creation.
        
        Args:
            agent: The agent instance
            file_path: Path to the file to be created
            
        Returns:
            str: Result of directory creation
        """
        try:
            # Get parent directory
            parent_dir = os.path.dirname(file_path)
            
            # Check if parent directory exists
            if not os.path.exists(parent_dir):
                # Create parent directories
                os.makedirs(parent_dir, exist_ok=True)
                return f"Created parent directories for: {file_path}"
            else:
                return f"Parent directories already exist for: {file_path}"
            
        except Exception as e:
            return f"Error creating parent directories: {str(e)}"
    
    def analyze_project_templates(self, agent: Agent, repo_path: str) -> str:
        """
        Analyze existing files to identify project templates and patterns.
        
        Args:
            agent: The agent instance
            repo_path: Path to the repository root
            
        Returns:
            str: Analysis of project templates and patterns
        """
        try:
            # Find persona patterns
            persona_patterns = []
            
            # Walk through the repository
            for root, dirs, files in os.walk(repo_path):
                # Skip common non-code directories
                if any(skip_dir in root for skip_dir in ['.git', 'node_modules', '__pycache__', '.venv']):
                    continue
                
                # Look for persona.py files
                if 'persona.py' in files:
                    rel_path = os.path.relpath(os.path.join(root, 'persona.py'), repo_path)
                    persona_patterns.append(rel_path)
            
            # Analyze persona structure
            persona_analysis = "Persona Structure Analysis:\n"
            
            for persona_path in persona_patterns:
                try:
                    full_path = os.path.join(repo_path, persona_path)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract class name
                    class_match = re.search(r'class\s+(\w+)\(BasePersona\)', content)
                    if class_match:
                        class_name = class_match.group(1)
                        
                        # Check for common files in the same directory
                        persona_dir = os.path.dirname(full_path)
                        dir_files = os.listdir(persona_dir)
                        
                        persona_analysis += f"\n- {class_name} ({persona_path}):\n"
                        persona_analysis += f"  Directory: {os.path.dirname(persona_path)}\n"
                        persona_analysis += f"  Files: {', '.join(dir_files)}\n"
                except:
                    # Skip files that can't be read
                    pass
            
            return persona_analysis
            
        except Exception as e:
            return f"Error analyzing project templates: {str(e)}"