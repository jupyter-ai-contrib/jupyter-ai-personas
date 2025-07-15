import re
import os
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

class DataAnalyticsTeam(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="DataAnalyticsTeam",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="An intelligent data analysis assistant that can chat, explain concepts, and perform comprehensive data analysis workflows including EDA, preprocessing, and visualization.",
            system_prompt="I am a data analysis team designed to help with comprehensive " 
                        "data analysis workflows. I coordinate specialized team members: " 
                        "an EDA agent who extracts and analyzes data, a preprocessor who " 
                        "cleans and organizes data, a code generator who creates " 
                        "visualization code, and a visualizer who executes and saves plots. " 
                        "Together, we provide complete data analysis pipelines with insights "
                        "and visualizations.",
        )

    def initialize_team(self, system_prompt: str, user_message: str):
        # Validate required configuration
        if not hasattr(self.config_manager, 'lm_provider_params') or 'model_id' not in self.config_manager.lm_provider_params:
            raise ValueError("Model ID not found in configuration")

        model_id = self.config_manager.lm_provider_params["model_id"]

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
                "",
                f"USER_MESSAGE = '''{user_message}'''",
                "",
                "PHASE 1 - DATA EXTRACTION FROM USER_MESSAGE:",
                "1. Print the user message first: print('USER MESSAGE TO ANALYZE:', USER_MESSAGE)",
                "2. Look ONLY for actual data in USER_MESSAGE - no creation of new data",
                "3. Extract data patterns EXACTLY as provided in USER_MESSAGE:",
                "   - DataFrame creation: df = pd.DataFrame(...) [extract exact values]",
                "   - Dictionary data: data = {...} [extract exact structure]", 
                "   - CSV text: comma-separated values [extract exact text]",
                "   - List/array data: [1,2,3] [extract exact values]",
                "4. Execute ONLY the extracted user data code from USER_MESSAGE",
                "5. NEVER add, modify, or supplement the user's data",
                "",
                "EXTRACTION EXAMPLES (use exact user values from USER_MESSAGE):",
                "- If USER_MESSAGE contains: 'df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]})'",
                "  Extract and execute: df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]}) [EXACT SAME]",
                "- If USER_MESSAGE contains raw CSV: 'name,age\\nJohn,25\\nJane,30'",
                "  Use: df = pd.read_csv(io.StringIO('name,age\\nJohn,25\\nJane,30')) [EXACT SAME]",
                "",
                "FORBIDDEN ACTIONS:",
                "   - NEVER use np.random or random to generate data",
                "   - NEVER create example values like [1, 2, 3, 4, 5]",
                "   - NEVER use pd.DataFrame() with values you invented",
                "   - NEVER create synthetic AWS employee data or any other fake data",
                "   - If USER_MESSAGE only has column names without values, STOP and report error",
                "",
                "EXTRACTION VERIFICATION:",
                "   - After extraction, print: print('Extracted data preview:', df.head())",
                "   - If df is empty or has no rows, extraction FAILED",
                "   - The data values in df must EXACTLY match what user provided in USER_MESSAGE",
                "",
                "PHASE 2 - EXPLORATORY DATA ANALYSIS:",
                "Only proceed with EDA if you successfully extracted real data from USER_MESSAGE:",
                "",
                "A. BASIC INFORMATION:",
                "   - Dataset shape, columns, data types",
                "   - First few rows to verify correct extraction",
                "",
                "B. DATA QUALITY ASSESSMENT:",
                "   - Missing values analysis",
                "   - Duplicate detection",
                "   - Unique value counts",
                "",
                "C. STATISTICAL SUMMARY:",
                "   - Descriptive statistics for numeric columns",
                "   - Frequency analysis for categorical columns",
                "",
                "PHASE 3 - SAVE ONLY IF REAL DATA FROM USER_MESSAGE:",
                "MAKE SURE to save the csv as 'extracted_data.csv'",
                "if len(df) > 0 and 'real data was extracted from USER_MESSAGE':",
                f"    df.to_csv(os.path.join(r'{abs_session_dir}', 'extracted_data.csv'), index=False)",
                "    print(f'Saved {{len(df)}} rows of USER-PROVIDED data from USER_MESSAGE')",
                "else:",
                "    print('ERROR: No real data to save - only column names found in USER_MESSAGE')",
                "",
                "Signal 'DATA_EXTRACTED' only if real user data from USER_MESSAGE was saved"
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
                "INSPECT ACTUAL COLUMNS:",
                "print(f'Actual columns in extracted data: {list(df.columns)}')",
                "print(f'Data types: {df.dtypes}')",
                "print(f'Data shape: {df.shape}')",
                "",
                "COLUMN STANDARDIZATION:",
                "# Standardize column names to lowercase with underscores",
                "original_columns = list(df.columns)",
                "df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')",
                "standardized_columns = list(df.columns)",
                "print(f'Original columns: {original_columns}')",
                "print(f'Standardized columns: {standardized_columns}')",
                "",
                "BASIC DATA CLEANING:",
                "print(f'Data before cleaning: {df.shape}')",
                "print(f'Missing values per column: {df.isnull().sum()}')",
                "df_cleaned = df.dropna().drop_duplicates()",
                "print(f'Data after cleaning: {df_cleaned.shape}')",
                "",
                "SAVE CLEANED DATA:",
                "df_cleaned.to_csv('cleaned_data.csv', index=False)",
                "print(f'cleaned_data.csv exists: {os.path.exists(\"cleaned_data.csv\")}')",
                "print(f'Final cleaned columns: {list(df_cleaned.columns)}')",
                "",
                "print(f'Files in directory: {os.listdir(\".\")}') ",
                "Document all preprocessing steps taken"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        visualizer_agent = Agent(
            name="visualization_agent",
            role="Data visualization specialist who generates and executes plots",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                f"SESSION_DIR = r'{abs_session_dir}'",
                "Work in SESSION_DIR directory",
                "",
                "SETUP:",
                "- Import required libraries (matplotlib, pandas, numpy)",
                "- NEVER use plt.style.use('seaborn') - it will cause an error",
                "- Load cleaned_data.csv and inspect its actual columns",
                "- Check available columns and data types before creating visualizations",
                "",
                "CRITICAL - INSPECT DATA FIRST:",
                "1. Load cleaned_data.csv",
                "2. Print actual column names: print(f'Available columns: {list(df.columns)}')",
                "3. Print data types: print(f'Data types: {df.dtypes}')",
                "4. Print data shape: print(f'Data shape: {df.shape}')",
                "5. Print sample data: print(df.head())",
                "",
                "VISUALIZATION WORKFLOW:",
                "",
                "1. CREATE VISUALIZATION CODE FILES BASED ON ACTUAL COLUMNS:",
                "   For each visualization you want to create:",
                "   a) Use ONLY the actual column names from cleaned_data.csv",
                "   b) Write the complete Python code as a string",
                "   c) Save it to a .py file with descriptive name",
                "   d) Example pattern:",
                "      ```",
                "      # First inspect the data",
                "      df = pd.read_csv('cleaned_data.csv')",
                "      actual_columns = list(df.columns)",
                "      numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()",
                "      categorical_columns = df.select_dtypes(include=['object']).columns.tolist()",
                "      ",
                "      # Then create visualizations using actual column names",
                "      code = '''",
                "      import matplotlib.pyplot as plt",
                "      import pandas as pd",
                "      import numpy as np",
                "      ",
                "      df = pd.read_csv('cleaned_data.csv')",
                "      ",
                "      # Use actual column names here",
                "      if 'column_name' in df.columns:",
                "          plt.figure(figsize=(10, 6))",
                "          plt.hist(df['column_name'], bins=30)",
                "          plt.title('Distribution of Column Name')",
                "          plt.xlabel('Value')",
                "          plt.ylabel('Frequency')",
                "          plt.savefig('distribution_plot.png', dpi=300, bbox_inches='tight')",
                "          plt.close()",
                "      '''",
                "      ",
                "      with open('01_distribution_plot.py', 'w') as f:",
                "          f.write(code)",
                "      ```",
                "",
                "2. MANDATORY FILE SAVING:",
                "   - ALWAYS save the complete Python code of the plots to .py files BEFORE execution",
                "   - Use descriptive filenames: 01_distribution_plots.py, 02_correlation_heatmap.py, etc.",
                "   - Each file must be self-contained and executable",
                "",
                " EXECUTION AFTER SAVING:",
                "   - Only after ALL .py files are saved, execute them to generate PNG files",
                "   - Verify that both .py and .png files exist in the directory",
                "",
                "3. ADAPTIVE VISUALIZATIONS (save as separate .py files):",
                "   Based on the actual data columns found:",
                "   - 01_distribution_plots.py: Histograms for numeric columns (if any)",
                "   - 02_correlation_heatmap.py: Correlation matrix if multiple numeric columns",
                "   - 03_categorical_analysis.py: Bar charts for categorical columns (if any)",
                "   - 04_scatter_plots.py: Relationships between numeric variables (if 2+ numeric)",
                "   - 05_summary_dashboard.py: Combined overview visualization",
                "",
                "4. FILE NAMING CONVENTION:",
                "   - Use numbered prefixes: 01_, 02_, etc.",
                "   - Descriptive names based on actual data",
                "   - Both .py files and resulting .png files",
                "",
                "5. FINAL OUTPUT:",
                "   - Python code files with actual column names",
                "   - Image files generated from actual data",
                "   - Summary of all created files",
                "",
                "IMPORTANT:",
                "- Each .py file should be self-contained and runnable",
                "- Include all necessary imports in each file",
                "- Use matplotlib only (no seaborn)",
                "- Save high-quality images (300 dpi)",
                "- NEVER hardcode column names - always use actual columns from the data",
                "",
                "Signal 'VISUALIZATIONS_COMPLETE' with list of created files"
            ],
            tools=[ImprovedPythonTools(session_dir=abs_session_dir), PandasTools()],
            markdown=True,
            show_tool_calls=True
        )

        data_analysis_team = Team(
            name="data-analysis-team",
            mode="coordinate",
            members=[eda_agent, preprocessor_agent, visualizer_agent],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                f"Chat history: " + system_prompt,
                "Coordinate a complete data analysis workflow from raw data to final visualizations",
                "",
                "WORKFLOW FOR DATA:",
                "1. EDA Agent: Extract data → save as extracted_data.csv", 
                "2. Preprocessor: Clean data → save as cleaned_data.csv",
                "3. Visualizer: Create plots → save as PNG files",
                "",
                "KEY PRINCIPLES:",
                "- Each agent depends on the previous agent's output",
                "- Verify previous outputs exist before proceeding",
                "- Use ONLY the actual data provided by the user",
                "- NEVER generate synthetic or example data",
                "- Provide insights and findings at each stage",
                "- Handle errors gracefully with clear feedback",
                "",
                "DELIVERABLES:",
                "- Extracted raw data (extracted_data.csv)",
                "- Cleaned, standardized data (cleaned_data.csv)",
                "- Professional visualizations (multiple PNG files)",
                "- Comprehensive insights throughout the process"
            ],
            markdown=True,
            show_members_responses=False,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=False
        )
        return data_analysis_team

    async def process_message(self, message: Message):
        message_text = message.body
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]

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
            "
            CRITICAL REQUIREMENTS:

            ALWAYS directly give your response and NOT your thought process.
            
            DATA EXTRACTION:
            - Extract ONLY the actual data from user message
            - NEVER generate synthetic data or examples
            - Use exact values provided by the user
            - Ignore imports, print statements, focus on data creation code
            
            COLUMN NAME STANDARDIZATION:
            - Standardize all column names to lowercase with underscores
            - Simple text-based column info (no JSON serialization issues)
            - Prevent 'Department' vs 'department' type errors
            
            AUTOMATIC VISUALIZATION STRATEGY:
            - Auto-detect column types (numeric vs text vs datetime)
            - Generate appropriate visualizations based on actual data types
            - Professional styling without seaborn
            - Adaptive analysis types based on available data:
              * Distribution Analysis (histograms for numeric data)
              * Relationship Analysis (correlation heatmaps)
              * Categorical Analysis (bar charts for text data)
              * Summary Dashboard (comprehensive overview)
            
            Expected input scenarios:
            - Mixed Python code with imports + data creation (extract only data parts)
            - Raw CSV data (comma-separated text)
            - Pure data creation code (DataFrames, dictionaries, lists)
            - JSON data with surrounding code
            - File paths with additional code
            
            Data extraction examples:
            - From: "import pandas as pd\\ndf = pd.DataFrame({{'x': [1,2,3]}})\\nprint('hello')" 
            - Extract: "df = pd.DataFrame({{'x': [1,2,3]}})"
            - Use all the given data values from the user's original input
            
            Context: {history_text}
            Model: {model_id} from {provider_name}
            "
            Always provide helpful responses with data insights, findings, and explanations of the analysis process.
            If any step fails, provide clear error messages and guidance to the user.
            """

        # Initialize and run the team with error handling
        data_team = self.initialize_team(system_prompt, message_text)

        # Pass the user message explicitly to ensure data extraction
        response = data_team.run(
            message_text,
            stream=False,
            stream_intermediate_steps=False,
            show_full_reasoning=False,
        )

        # Extract key insights for user-friendly response
        response_content = response.content

        async def response_iterator():
            yield response_content

        await self.stream_message(response_iterator())