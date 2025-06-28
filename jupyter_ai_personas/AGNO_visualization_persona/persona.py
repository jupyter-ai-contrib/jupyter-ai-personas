import os
import re
import logging
import boto3
import datetime
from pathlib import Path
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory

from langchain_core.messages import HumanMessage
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.pandas import PandasTools
from .enhancedPythonTools import ImprovedPythonTools

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS session with error handling
try:
    session = boto3.Session()
    session.get_credentials()
except Exception as e:
    logger.error(f"AWS credentials not configured: {e}")
    session = None

def create_timestamped_session_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}"

SESSION_DIR = create_timestamped_session_dir()

class VisualizationAssistant(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="VisualizationAssistant",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized data analysis team that performs EDA, preprocessing, visualization generation, and plot creation.",
            system_prompt="I am a data analysis team designed to help with comprehensive data analysis workflows. I coordinate specialized team members: an EDA agent who extracts and analyzes data, a preprocessor who cleans and organizes data, a code generator who creates visualization code, and a visualizer who executes and saves plots. Together, we provide complete data analysis pipelines with insights and visualizations.",
        )
    
    def initialize_team(self, system_prompt):
        # Validate required configuration
        if not hasattr(self.config, 'lm_provider_params') or 'model_id' not in self.config.lm_provider_params:
            raise ValueError("Model ID not found in configuration")
        
        model_id = self.config.lm_provider_params["model_id"]
        
        if not session:
            raise ValueError("AWS session not properly configured")
        
        # Create single directory for all files with absolute path
        abs_session_dir = os.path.abspath(SESSION_DIR)
        os.makedirs(abs_session_dir, exist_ok=True)
        logger.info(f"Working directory: {abs_session_dir}")
        
        eda_agent = Agent(
            name="eda_agent",
            role="Exploratory Data Analysis specialist who extracts and analyzes data",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "CRITICAL: Start every task by running this setup code:",
                f"import os",
                f"import pandas as pd",
                f"import numpy as np",
                f"import io",
                f"import re",
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"print(f'Session directory created/verified: {{SESSION_DIR}}')",
                f"print(f'Current working directory before change: {{os.getcwd()}}')",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                f"print(f'Files in session directory: {{os.listdir(\".\")}}')",
                "",
                "NEVER CREATE SYNTHETIC DATA, use ONLY the user's provided data - NO exceptions",
                "",
                "MANDATORY DATA EXTRACTION:",
                "1. Print the ENTIRE user message first: print('USER MESSAGE:', message_text)",
                "2. Look ONLY for actual data in the message - no creation of new data",
                "3. Extract data patterns EXACTLY as provided:",
                "   - DataFrame creation: df = pd.DataFrame(...) [extract exact values]",
                "   - Dictionary data: data = {...} [extract exact structure]", 
                "   - CSV text: comma-separated values [extract exact text]",
                "   - List/array data: [1,2,3] [extract exact values]",
                "4. Execute ONLY the extracted user data code",
                "5. NEVER add, modify, or supplement the user's data",
                "",
                "EXTRACTION EXAMPLES (use exact user values):",
                "- From: 'df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]})'",
                "  Extract: df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]}) [EXACT SAME]",
                "- From raw CSV: 'name,age\\nJohn,25\\nJane,30'",
                "  Use: pd.read_csv(io.StringIO('name,age\\nJohn,25\\nJane,30')) [EXACT SAME]",
                "",
                "CRITICAL FILE SAVING - Use absolute path to prevent duplicates:",
                f"extracted_data_path = os.path.join(r'{abs_session_dir}', 'extracted_data.csv')",
                "df.to_csv(extracted_data_path, index=False)",
                "print(f'User data saved to: {extracted_data_path}')",
                "print(f'File exists at correct location: {os.path.exists(extracted_data_path)}')",
                "print(f'File size: {os.path.getsize(extracted_data_path)} bytes')",
                "",
                "Coordinate a data analysis workflow: Extract → Clean → Generate → Visualize",
                f"All files must be saved to: {abs_session_dir}",
                "Each agent depends on the previous agent's output",
                "Provide insights and findings throughout the process",
                "NEVER generate random or sample data",
                "",
                "ONLY proceed if user provided actual data to extract"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        preprocessor_agent = Agent(
            name="preprocessor_agent",
            role="Data preprocessing specialist who cleans and organizes data",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "CRITICAL: Start every task by running this setup code:",
                f"import os",
                f"import pandas as pd",
                f"import numpy as np",
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "FIRST: Check if extracted_data.csv exists: print(f'extracted_data.csv exists: {os.path.exists(\"extracted_data.csv\")}')",
                "If file doesn't exist, report error and ask EDA agent to extract data first",
                "Load the extracted data: df = pd.read_csv('extracted_data.csv')",
                "",
                "SIMPLIFIED COLUMN STANDARDIZATION:",
                "print(f'Original columns: {list(df.columns)}')",
                "df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')",
                "print(f'Standardized columns: {list(df.columns)}')",
                "",
                "BASIC DATA CLEANING:",
                "df = df.dropna().drop_duplicates()",
                "print(f'Cleaned data shape: {df.shape}')",
                "",
                "SAVE CLEANED DATA:",
                "df.to_csv('cleaned_data.csv', index=False)",
                "print(f'cleaned_data.csv exists: {os.path.exists(\"cleaned_data.csv\")}')",
                "",
                "SAVE SIMPLE COLUMN INFO (NO JSON SERIALIZATION ISSUES):",
                "with open('column_info.txt', 'w') as f:",
                "    f.write(f'columns: {list(df.columns)}\\n')",
                "    f.write(f'shape: {list(df.shape)}\\n')",
                "    f.write(f'numeric: {list(df.select_dtypes(include=[\"number\"]).columns)}\\n')",
                "    f.write(f'text: {list(df.select_dtypes(include=[\"object\"]).columns)}\\n')",
                "print('Column info saved to column_info.txt')",
                "",
                "print(f'Files in directory: {os.listdir(\".\")}') ",
                "Document all preprocessing steps taken"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        code_generator_agent = Agent(
            name="code_generator_agent",
            role="Visualization code generator",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "CRITICAL: Start every task by running this setup code:",
                f"import os",
                f"import pandas as pd",
                f"import numpy as np",
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "LOAD DATA AND DETECT COLUMN TYPES:",
                f"df = pd.read_csv('cleaned_data.csv')",
                "print(f'Data shape: {df.shape}')",
                "print(f'Columns: {list(df.columns)}')",
                "",
                "# Simple column type detection - no complex JSON loading needed",
                "numeric_cols = list(df.select_dtypes(include=['number']).columns)",
                "text_cols = list(df.select_dtypes(include=['object']).columns)",
                "datetime_cols = list(df.select_dtypes(include=['datetime']).columns)",
                "print(f'Numeric columns: {numeric_cols}')",
                "print(f'Text columns: {text_cols}')",
                "print(f'DateTime columns: {datetime_cols}')",
                "",
                "CREATE VISUALIZATION FUNCTIONS:",
                "def save_viz_code(filename, viz_code):",
                f"    full_path = os.path.join(r'{abs_session_dir}', f'{{filename}}.py')",
                "    with open(full_path, 'w') as f:",
                "        f.write(viz_code)",
                "    print(f'Created: {full_path}')",
                "    return os.path.exists(full_path)",
                "",
                "GENERATE VISUALIZATION SCRIPTS",
                "",
                "# VERIFY CREATED FILES",
                f"py_files = [f for f in os.listdir(r'{abs_session_dir}') if f.endswith('.py')]",
                "print(f'Created {len(py_files)} visualization scripts: {py_files}')",
                "for py_file in py_files:",
                f"    file_path = os.path.join(r'{abs_session_dir}', py_file)",
                "    if os.path.exists(file_path):",
                "        size = os.path.getsize(file_path)",
                "        print(f'✅ {py_file}: {size} bytes')",
                "    else:",
                "        print(f'❌ {py_file}: not found')"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        visualizer_agent = Agent(
            name="visualizer_agent",
            role="Plot execution specialist who runs code and saves visualization images",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "CRITICAL: Start every task by running this setup code:",
                f"import os",
                f"import subprocess",
                f"import sys",
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "Find Python plot files: python_files = [f for f in os.listdir('.') if f.endswith('.py')]",
                "Execute each Python file in current directory:",
                "for py_file in python_files:",
                "    try:",
                "        print(f'Executing {py_file}...')",
                "        result = subprocess.run([sys.executable, py_file], capture_output=True, text=True)",
                "        if result.returncode == 0:",
                "            print(f'✅ {py_file} executed successfully')",
                "            if result.stdout:",
                "                print(f'Output: {result.stdout.strip()}')",
                "        else:",
                "            print(f'❌ {py_file} failed: {result.stderr}')",
                "    except Exception as e:",
                "        print(f'❌ Error executing {py_file}: {e}')",
                "",
                "print(f'\\n🎯 VISUALIZATION SUMMARY:')",
                "print(f'Created {len(image_files)} high-quality visualizations')",
                "print(f'Total file sizes: {sum(os.path.getsize(f) for f in image_files):,} bytes')",
                "Generate a summary of all created visualizations with their file paths",
                "Handle errors gracefully and provide debugging information"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir)],
            markdown=True,
            show_tool_calls=True
        )

        data_analysis_team = Team(
            name="data-analysis-team",
            mode="coordinate",
            members=[eda_agent, preprocessor_agent, code_generator_agent, visualizer_agent],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Chat history: " + system_prompt,
                "Coordinate a complete data analysis workflow from raw data to final visualizations",
                f"SINGLE DIRECTORY: All files must be saved to '{abs_session_dir}' directory",
                "File flow: extracted_data.csv → cleaned_data.csv → plot_codes.py → plot_images.png",
                "EDA agent MUST create extracted_data.csv in session directory before other agents proceed",
                "",
                "SIMPLIFIED FEATURES:",
                "1. Column name standardization - lowercase with underscores",
                "2. Automatic visualization selection based on data types",
                "3. Professional styling and high-quality plot generation",
                "4. No complex JSON serialization - simple text-based column info",
                "",
                "Follow this strict sequence: EDA → Preprocessing → Code Generation → Visualization",
                "Each agent must verify the previous agent's output file exists in session directory before proceeding",
                "If any file is missing from session directory, stop and request the previous agent to complete their task",
                f"All agents must save their outputs to the same {abs_session_dir} directory",
                "Create organized output with all files in one centralized location",
                "Provide comprehensive insights and findings at each stage",
                "Share insights with the user throughout the process",
                "Handle errors gracefully and provide clear feedback about missing files",
                "Ensure all outputs are properly saved to the session directory and accessible",
                "If any agent fails, provide detailed error information and recovery suggestions"
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=True
        )
        return data_analysis_team

    async def process_message(self, message: Message):
        message_text = message.body
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]
            
        # Get conversation history with error handling
        try:
            history = YChatHistory(ychat=self.ychat, k=3)
            messages = await history.aget_messages()
        except Exception as e:
            logger.warning(f"Could not retrieve chat history: {e}")
            messages = []

        history_text = ""
        if messages:
            history_text = "\nPrevious conversation:\n"
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"

        # Create system prompt with context and data extraction guidance
        system_prompt = f"""
            You are coordinating a data analysis team in JupyterLab. Your goal is to:
            1. Extract and analyze data from user input (handle mixed code with imports, prints, and data)
            2. Clean and preprocess the data with proper column name standardization
            3. Generate intelligent, high-quality visualization code based on data characteristics
            4. Create and save professional plot images with enhanced styling
            5. Provide valuable insights and findings to the user
            
            SIMPLIFIED FEATURES IMPLEMENTED:
            
            🔧 COLUMN NAME STANDARDIZATION:
            - Standardize all column names to lowercase with underscores
            - Simple text-based column info (no JSON serialization issues)
            - Prevent 'Department' vs 'department' type errors
            
            🎨 AUTOMATIC VISUALIZATION STRATEGY:
            - Auto-detect column types (numeric vs text vs datetime)
            - Generate appropriate visualizations based on data types
            - Professional styling with seaborn themes
            - Four standard analysis types:
              * Distribution Analysis (histograms for numeric data)
              * Relationship Analysis (correlation heatmaps)
              * Categorical Analysis (bar charts for text data)
              * Summary Dashboard (comprehensive overview)
            
            CRITICAL: The EDA agent must successfully extract ONLY the data from mixed code input and save it as '{SESSION_DIR}/extracted_data.csv' 
            before any other agents can proceed. The agent should ignore imports, print statements, and focus on data extraction.
            
            Expected input scenarios:
            - Mixed Python code with imports + data creation (extract only data parts)
            - Raw CSV data (comma-separated text)
            - Pure data creation code (DataFrames, dictionaries, lists)
            - JSON data with surrounding code
            - File paths with additional code
            
            Data extraction examples:
            - From: "import pandas as pd\\ndf = pd.DataFrame({{'x': [1,2,3]}})\\nprint('hello')" 
            - Extract: "df = pd.DataFrame({{'x': [1,2,3]}})"
            
            Context: {history_text}
            Model: {model_id} from {provider_name}
            
            Always provide helpful responses with data insights, findings, and explanations of the analysis process.
            If any step fails, provide clear error messages and guidance to the user.
            """

        # Initialize and run the team with error handling
        data_team = self.initialize_team(system_prompt)
                
        response = data_team.run(
            message_text,
            stream=False,
            stream_intermediate_steps=False,
            show_full_reasoning=True,
        )
                
        # Extract key insights for user-friendly response
        response_content = response.content
                
        async def response_iterator():
            yield response_content
        
        await self.stream_message(response_iterator())
        