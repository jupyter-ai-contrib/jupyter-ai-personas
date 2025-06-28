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

class DataAnalyticsTeam(BasePersona):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="DataAnalyticsTeam",
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
                "",
                "NEVER GENERATE DATA OTHER THAN THE DATA PROVIDED BY THE USER"
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
                "  EXTRACTION EXAMPLES (use exact user values):",
                "- From: 'df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]})'",
                "  Extract: df = pd.DataFrame({\"x\": [1,2,3], \"y\": [4,5,6]}) [EXACT SAME]",
                "- From raw CSV: 'name,age\\nJohn,25\\nJane,30'",
                "  Use: pd.read_csv(io.StringIO('name,age\\nJohn,25\\nJane,30')) [EXACT SAME]",
                "",
                "3. FORBIDDEN ACTIONS:",
                "   - NEVER use np.random or random to generate data",
                "   - NEVER create example values like [1, 2, 3, 4, 5]",
                "   - NEVER use pd.DataFrame() with values you invented",
                "   - If you only see column names without values, STOP and report error",
                "",
                "4. EXTRACTION VERIFICATION:",
                "   - After extraction, print: print('Extracted data preview:', df.head())",
                "   - If df is empty or has no rows, extraction FAILED",
                "   - The data values in df must EXACTLY match what user provided",
                "",
                "PHASE 2 - EXPLORATORY DATA ANALYSIS:",
                "Only proceed with EDA if you successfully extracted real data:",
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
                "PHASE 3 - SAVE ONLY IF REAL DATA:",
                "if len(df) > 0 and 'real data was extracted':",
                f"    df.to_csv(os.path.join(r'{abs_session_dir}', 'extracted_data.csv'), index=False)",
                "    print(f'Saved {len(df)} rows of USER-PROVIDED data')",
                "else:",
                "    print('ERROR: No real data to save - only column names found')",
                "",
                "Signal 'DATA_EXTRACTED' only if real user data was saved"
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
                "- Load cleaned_data.csv",
                "- Check available columns and data types",
                "",
                "VISUALIZATION WORKFLOW:",
                "",
                "1. CREATE VISUALIZATION CODE FILES:",
                "   For each visualization you want to create:",
                "   a) Write the complete Python code as a string",
                "   b) Save it to a .py file with descriptive name",
                "   c) Example pattern:",
                "      ```",
                "      code = '''",
                "      import matplotlib.pyplot as plt",
                "      import pandas as pd",
                "      ",
                "      df = pd.read_csv('cleaned_data.csv')",
                "      plt.figure(figsize=(10, 6))",
                "      plt.hist(df['column_name'], bins=30)",
                "      plt.title('Distribution of Column Name')",
                "      plt.xlabel('Value')",
                "      plt.ylabel('Frequency')",
                "      plt.savefig('distribution_plot.png', dpi=300, bbox_inches='tight')",
                "      plt.close()",
                "      '''",
                "      ",
                "      with open('01_distribution_plot.py', 'w') as f:",
                "          f.write(code)",
                "      ```",
                "",
                "2. EXECUTE EACH VISUALIZATION SCRIPT:",
                "   After saving all .py files:",
                "   - Use exec() or subprocess to run each script",
                "   - This will generate the PNG files",
                "",
                "3. REQUIRED VISUALIZATIONS (save as separate .py files):",
                "   - 01_distribution_plots.py: Histograms for numeric columns",
                "   - 02_correlation_heatmap.py: Correlation matrix if multiple numeric columns",
                "   - 03_categorical_analysis.py: Bar charts for categorical columns",
                "   - 04_scatter_plots.py: Relationships between numeric variables",
                "   - 05_summary_dashboard.py: Combined overview visualization",
                "",
                "4. FILE NAMING CONVENTION:",
                "   - Use numbered prefixes: 01_, 02_, etc.",
                "   - Descriptive names: distribution_plots, correlation_heatmap",
                "   - Both .py files and resulting .png files",
                "",
                "5. FINAL OUTPUT:",
                "   - Python code files: 01_distribution_plots.py, 02_correlation_heatmap.py, etc.",
                "   - Image files: distribution_plot_1.png, correlation_heatmap.png, etc.",
                "   - Summary of all created files",
                "",
                "IMPORTANT:",
                "- Each .py file should be self-contained and runnable",
                "- Include all necessary imports in each file",
                "- Use matplotlib only (no seaborn)",
                "- Save high-quality images (300 dpi)",
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
                "Chat history: " + system_prompt,
                "Coordinate a complete data analysis workflow from raw data to final visualizations",
                f"All files must be saved to '{abs_session_dir}' directory",
                "",
                "WORKFLOW:",
                "1. EDA Agent: Extract data from user input → save as extracted_data.csv",
                "2. Preprocessor: Clean and standardize data → save as cleaned_data.csv", 
                "3. Visualizer: Create and save visualization plots → save as PNG files",
                "",
                "KEY PRINCIPLES:",
                "- Each agent depends on the previous agent's output",
                "- Verify previous outputs exist before proceeding",
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
            
            COLUMN NAME STANDARDIZATION:
            - Standardize all column names to lowercase with underscores
            - Simple text-based column info (no JSON serialization issues)
            - Prevent 'Department' vs 'department' type errors
            
            AUTOMATIC VISUALIZATION STRATEGY:
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
        