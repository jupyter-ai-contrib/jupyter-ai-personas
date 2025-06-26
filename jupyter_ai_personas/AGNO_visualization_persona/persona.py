from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
import boto3
from langchain_core.messages import HumanMessage
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.pandas import PandasTools
import os
import re
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS session with error handling
try:
    session = boto3.Session()
    # Verify AWS credentials are available
    session.get_credentials()
except Exception as e:
    logger.error(f"AWS credentials not configured: {e}")
    session = None

# Define consistent single directory for all files
SESSION_DIR = "session_visualizer"

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
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "CRITICAL: The user's input data will be provided in the message. You MUST extract and use ONLY the user's actual data.",
                "DO NOT CREATE SYNTHETIC DATA. DO NOT GENERATE FAKE DATA. USE ONLY THE USER'S PROVIDED DATA.",
                "",
                "DATA EXTRACTION PROCESS:",
                "1. First, print the entire user message to understand what data was provided",
                "2. Look for these data patterns in the user's message:",
                "   - DataFrame creation: df = pd.DataFrame(...)",
                "   - Dictionary data: data = {...} or records = [...]", 
                "   - CSV text: comma-separated values in text format",
                "   - List/array data: [1,2,3] or similar structures",
                "   - JSON data: {...} objects",
                "3. Extract ONLY the data-creation code, ignore imports/prints/comments",
                "4. Execute the extracted data code to create the DataFrame",
                "5. If user provides raw CSV text, use pd.read_csv(io.StringIO(csv_text))",
                "",
                "EXAMPLES:",
                "- From: 'import pandas as pd\\ndf = pd.DataFrame({\"x\": [1,2,3]})\\nprint(df)'",
                "  Extract: 'df = pd.DataFrame({\"x\": [1,2,3]})'",
                "- From raw CSV: 'name,age\\nJohn,25\\nJane,30'",
                "  Use: pd.read_csv(io.StringIO('name,age\\nJohn,25\\nJane,30'))",
                "",
                "AFTER EXTRACTING USER DATA:",
                "- Verify the DataFrame was created: print(f'DataFrame shape: {df.shape}')",
                "- Show first few rows: print(df.head())",
                "- Perform comprehensive EDA on the USER'S ACTUAL DATA",
                "- MANDATORY: Save the user's data: df.to_csv('extracted_data.csv', index=False)",
                "- Verify file creation: print(f'File exists: {os.path.exists(\"extracted_data.csv\")}')",
                "- List files: print(f'Files in directory: {os.listdir(\".\")}') ",
                "",
                "If you cannot find any data in the user's message, ask them to provide data before proceeding.",
                "DO NOT make up data. DO NOT use sample/synthetic data."
            ],
            tools=[PythonTools(), PandasTools()],
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
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "FIRST: Check if extracted_data.csv exists: print(f'extracted_data.csv exists: {os.path.exists(\"extracted_data.csv\")}')",
                "If file doesn't exist, report error and ask EDA agent to extract data first",
                "Load the extracted data: df = pd.read_csv('extracted_data.csv')",
                "Clean and preprocess the data (handle missing values, duplicates, outliers)",
                "Standardize column names and data formats",
                "Perform data type conversions as needed",
                "Handle categorical encoding if necessary",
                "MANDATORY: Save cleaned data: df.to_csv('cleaned_data.csv', index=False)",
                "Verify file creation: print(f'cleaned_data.csv exists: {os.path.exists(\"cleaned_data.csv\")}')",
                "List files: print(f'Files in directory: {os.listdir(\".\")}') ",
                "Document all preprocessing steps taken",
                "Ensure data quality and consistency",
                "Use pandas for efficient data cleaning and transformation operations",
                "Handle errors gracefully and report if preprocessing fails"
            ],
            tools=[PythonTools(), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        code_generator_agent = Agent(
            name="code_generator_agent",
            role="Visualization code generator who creates plot and analysis code",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "CRITICAL: Start every task by running this setup code:",
                f"import os",
                f"import pandas as pd",
                f"SESSION_DIR = r'{abs_session_dir}'",
                f"os.makedirs(SESSION_DIR, exist_ok=True)",
                f"os.chdir(SESSION_DIR)",
                f"print(f'Working in directory: {{os.getcwd()}}')",
                "",
                "FIRST: Check if cleaned_data.csv exists: print(f'cleaned_data.csv exists: {os.path.exists(\"cleaned_data.csv\")}')",
                "If file doesn't exist, report error and ask preprocessor agent to clean data first",
                "",
                "CRITICAL: You must ACTUALLY CREATE AND SAVE .py files, not just generate code content!",
                "",
                "PROCESS:",
                "1. Load the cleaned data to understand its structure: df = pd.read_csv('cleaned_data.csv')",
                "2. Print data info: print(f'Data shape: {df.shape}, Columns: {list(df.columns)}')",
                "3. For each visualization type, CREATE the code AND SAVE it to a .py file",
                "",
                "EXAMPLE - Create histogram analysis:",
                "histogram_code = '''",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "",
                "# Load data",
                "df = pd.read_csv('cleaned_data.csv')",
                "",
                "# Create histogram",
                "plt.figure(figsize=(10, 6))",
                "df.hist(bins=20, figsize=(15, 10))",
                "plt.suptitle('Data Distribution Analysis')",
                "plt.tight_layout()",
                "plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')",
                "plt.close()",
                "print('Histogram saved as histogram_analysis.png')",
                "'''",
                "",
                "# SAVE the code to file",
                "with open('histogram_analysis.py', 'w') as f:",
                "    f.write(histogram_code)",
                "print('Created: histogram_analysis.py')",
                "",
                "MANDATORY: Create at least 3-5 different visualization files:",
                "- histogram_analysis.py (distribution analysis)",
                "- correlation_heatmap.py (correlation matrix)",
                "- scatter_plots.py (relationships between variables)",
                "- box_plots.py (outlier detection)",
                "- summary_stats.py (statistical summaries)",
                "",
                "For each visualization:",
                "1. Create the code as a string variable",
                "2. Use with open('filename.py', 'w') as f: f.write(code_string)",
                "3. Print confirmation: print(f'Created: filename.py')",
                "",
                "VERIFY: List all created Python files:",
                "py_files = [f for f in os.listdir('.') if f.endswith('.py')]",
                "print(f'Created Python files: {py_files}')",
                "print(f'Total Python files: {len(py_files)}')",
                "",
                "Each generated .py file must:",
                "- Import necessary libraries (pandas, matplotlib, seaborn)",
                "- Load data with: df = pd.read_csv('cleaned_data.csv')",
                "- Create meaningful visualizations",
                "- Save plots as high-quality images (.png)",
                "- Include descriptive titles and labels",
                "- Use plt.close() to free memory",
                "",
                "Handle errors gracefully and report if code generation fails"
            ],
            tools=[PythonTools(), PandasTools()],
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
                "Print found files: print(f'Found Python files: {python_files}')",
                "Execute each Python file in current directory:",
                "for py_file in python_files:",
                "    try:",
                "        print(f'Executing {py_file}...')",
                "        result = subprocess.run([sys.executable, py_file], capture_output=True, text=True)",
                "        if result.returncode == 0:",
                "            print(f'✅ {py_file} executed successfully')",
                "        else:",
                "            print(f'❌ {py_file} failed: {result.stderr}')",
                "    except Exception as e:",
                "        print(f'❌ Error executing {py_file}: {e}')",
                "",
                "List all created files: print(f'All files in directory: {os.listdir(\".\")}') ",
                "Find image files: image_files = [f for f in os.listdir('.') if f.endswith(('.png', '.jpg', '.svg', '.pdf'))]",
                "Print created images: print(f'Created visualizations: {image_files}')",
                "Generate a summary of all created visualizations with their file paths",
                "Handle errors gracefully and provide debugging information"
            ],
            tools=[PythonTools()],
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
            2. Clean and preprocess the data
            3. Generate insightful visualization code
            4. Create and save plot images
            5. Provide valuable insights and findings to the user
            
            CRITICAL: The EDA agent must successfully extract ONLY the data from mixed code input and save it as 'session_visualizer/extracted_data.csv' 
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