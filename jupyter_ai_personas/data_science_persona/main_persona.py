import os
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3

# Your existing imports
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory

class EDAState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    has_data: bool
    dataframe: Optional[str]
    eda_complete: bool
    visualization_complete: bool
    next_action: str

class DataSciencePersona(BasePersona):
    """
    Simplified LangGraph persona focused on EDA and visualization only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plots_dir = "./plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize AWS session
        self.session = boto3.Session()
        model_id = self.config.lm_provider_params["model_id"]
        
        # Create the simplified workflow
        self.workflow = self._create_workflow(model_id)
        
    @property
    def defaults(self):
        return PersonaDefaults(
            name="DataSciencePersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Simplified EDA and visualization with LangGraph control.",
            system_prompt="I perform exploratory data analysis and create visualizations from user data.",
        )

    def _create_workflow(self, model_id: str):
        """Create simplified EDA workflow"""
        
        # Initialize the model
        llm = ChatBedrock(
            model_id=model_id,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.1}
        )
        
        # Build the graph
        workflow = StateGraph(EDAState)
        
        # Add nodes
        workflow.add_node("supervisor", self._create_supervisor(llm))
        workflow.add_node("data_processor", self._create_data_processor(llm))
        workflow.add_node("eda_analyst", self._create_eda_analyst(llm))
        workflow.add_node("visualizer", self._create_visualizer(llm))
        
        # Routing logic
        def route_supervisor(state: EDAState):
            next_action = state.get("next_action", "")
            
            if next_action == "process_data":
                return "data_processor"
            elif next_action == "analyze_data":
                return "eda_analyst"
            elif next_action == "create_visualizations":
                return "visualizer"
            else:
                return END
        
        # Add edges
        workflow.add_conditional_edges("supervisor", route_supervisor)
        workflow.add_edge("data_processor", "supervisor")
        workflow.add_edge("eda_analyst", "supervisor") 
        workflow.add_edge("visualizer", "supervisor")
        
        # Set entry point
        workflow.set_entry_point("supervisor") 
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _create_supervisor(self, llm):
        """Simple supervisor for EDA workflow"""
        
        def supervisor_node(state: EDAState):
            # Simple routing logic
            if not state.get("has_data", False):
                next_action = "process_data"
            elif not state.get("eda_complete", False):
                next_action = "analyze_data"
            elif not state.get("visualization_complete", False):
                next_action = "create_visualizations"
            else:
                next_action = "complete"
            
            return {
                **state,
                "next_action": next_action
            }
        
        return supervisor_node

    def _create_data_processor(self, llm):
        """Extract and process user data"""
        
        data_prompt = ChatPromptTemplate.from_messages([
            ("system", """You extract data from user input and convert it to a pandas DataFrame.
            
            Analyze the user input and:
            1. If it contains data structures (JSON, Python dicts/lists, CSV), extract them
            2. Convert to a clean pandas DataFrame
            3. If no data found, ask user to provide data
            
            Return either:
            - "DATA_EXTRACTED: [brief description]" if successful
            - "NO_DATA: [explanation]" if no data found"""),
            ("human", "Extract data from: {user_input}")
        ])
        
        def data_processor_node(state: EDAState):
            user_input = state.get("user_input", "")
            
            try:
                # Try to extract data using simple methods
                df = self._extract_data_simple(user_input)
                
                if df is not None and not df.empty:
                    # Convert DataFrame to JSON for storage in state
                    df_json = df.to_json(orient='records')
                    
                    return {
                        **state,
                        "has_data": True,
                        "dataframe": df_json,
                        "messages": state["messages"] + [AIMessage(content=f"Data extracted: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}")]
                    }
                else:
                    return {
                        **state,
                        "has_data": False,
                        "messages": state["messages"] + [AIMessage(content="No data found. Please provide data in JSON, CSV, or Python format.")]
                    }
                    
            except Exception as e:
                return {
                    **state,
                    "has_data": False,
                    "messages": state["messages"] + [AIMessage(content=f"Data processing error: {str(e)}")]
                }
        
        return data_processor_node

    def _create_eda_analyst(self, llm):
        """Perform exploratory data analysis"""
        
        eda_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an EDA specialist. Analyze the provided DataFrame and create a comprehensive exploratory data analysis report.
            
            Include:
            1. Data overview (shape, columns, types)
            2. Missing values analysis
            3. Statistical summaries for numeric columns
            4. Value distributions for categorical columns
            5. Key insights and patterns
            6. Recommendations for visualizations
            
            Be thorough but concise."""),
            ("human", "Perform EDA on this data: {data_info}")
        ])
        
        def eda_analyst_node(state: EDAState):
            df_json = state.get("dataframe")
            
            if not df_json:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="No data available for EDA")]
                }
            
            try:
                # Reconstruct DataFrame from JSON
                df = pd.read_json(df_json, orient='records')
                
                # Perform EDA
                eda_report = self._perform_eda(df)
                
                # Get LLM insights
                data_info = f"Shape: {df.shape}, Columns: {list(df.columns)}, Data types: {df.dtypes.to_dict()}"
                llm_response = llm.invoke(eda_prompt.format_messages(data_info=data_info))
                
                # Combine automated EDA with LLM insights
                full_report = f"""## 📊 Exploratory Data Analysis

                                {eda_report}

                                ## 🤖 AI Insights
                                {llm_response.content}
                                """
                
                return {
                    **state,
                    "eda_complete": True,
                    "messages": state["messages"] + [AIMessage(content=full_report)]
                }
                
            except Exception as e:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=f"EDA error: {str(e)}")]
                }
        
        return eda_analyst_node

    def _create_visualizer(self, llm):
        """Create visualizations"""
        
        viz_prompt = ChatPromptTemplate.from_messages([
            ("system", """You create appropriate visualizations based on the data analysis.
            
            Based on the data types and EDA results, suggest and create:
            1. Distribution plots for numeric variables
            2. Bar charts for categorical variables  
            3. Correlation heatmaps if multiple numeric variables
            4. Scatter plots for relationships
            
            Focus on the most informative visualizations."""),
            ("human", "Create visualizations for: {data_summary}")
        ])
        
        def visualizer_node(state: EDAState):
            df_json = state.get("dataframe")
            
            if not df_json:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content="No data available for visualization")]
                }
            
            try:
                # Reconstruct DataFrame
                df = pd.read_json(df_json, orient='records')
                
                # Create visualizations
                plots_created = self._create_visualizations(df)
                
                viz_summary = f"""## 📈 Visualizations Created

                **{len(plots_created)} plots generated:**
                {chr(10).join([f"- {plot}" for plot in plots_created])}

                **Saved to:** `{self.plots_dir}`

                The visualizations show the key patterns and distributions in your data. You can view the PNG files in the plots directory.
                """
                
                return {
                    **state,
                    "visualization_complete": True,
                    "messages": state["messages"] + [AIMessage(content=viz_summary)]
                }
                
            except Exception as e:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=f"Visualization error: {str(e)}")]
                }
        
        return visualizer_node

    def _extract_data_simple(self, user_input: str) -> pd.DataFrame:
        """Simple data extraction logic"""
        
        # Try JSON
        try:
            import json
            data = json.loads(user_input.strip())
            return pd.DataFrame(data)
        except:
            pass
        
        # Try executing as Python code
        try:
            # Safe execution environment
            env = {'pd': pd, 'pandas': pd, 'np': np, 'numpy': np, '__builtins__': {}}
            exec(user_input, env)
            
            # Look for DataFrames or data structures
            for var_name, var_value in env.items():
                if isinstance(var_value, pd.DataFrame) and not var_value.empty:
                    return var_value
                elif isinstance(var_value, dict):
                    try:
                        return pd.DataFrame(var_value)
                    except:
                        continue
                elif isinstance(var_value, list) and len(var_value) > 0:
                    try:
                        return pd.DataFrame(var_value)
                    except:
                        continue
        except:
            pass
        
        # Try CSV
        try:
            from io import StringIO
            if '\n' in user_input and (',' in user_input or '\t' in user_input):
                separator = '\t' if '\t' in user_input else ','
                return pd.read_csv(StringIO(user_input), sep=separator)
        except:
            pass
        
        return None

    def _perform_eda(self, df: pd.DataFrame) -> str:
        """Automated EDA report generation"""
        
        report = []
        
        # Basic info
        report.append(f"### Dataset Overview")
        report.append(f"- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        report.append(f"- **Columns:** {', '.join(df.columns)}")
        report.append("")
        
        # Data types
        report.append("### Data Types")
        for col, dtype in df.dtypes.items():
            report.append(f"- **{col}:** {dtype}")
        report.append("")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            report.append("### Missing Values")
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    report.append(f"- **{col}:** {count} ({pct:.1f}%)")
        else:
            report.append("### No Missing Values")
        report.append("")
        
        # Numeric summaries
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report.append("### Numeric Variables Summary")
            for col in numeric_cols:
                stats = df[col].describe()
                report.append(f"**{col}:**")
                report.append(f"  - Mean: {stats['mean']:.2f}")
                report.append(f"  - Median: {stats['50%']:.2f}")
                report.append(f"  - Std: {stats['std']:.2f}")
                report.append(f"  - Range: {stats['min']:.2f} to {stats['max']:.2f}")
                report.append("")
        
        # Categorical summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            report.append("### Categorical Variables Summary")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                report.append(f"**{col}:**")
                report.append(f"  - Unique values: {unique_count}")
                report.append(f"  - Top values: {', '.join([f'{val} ({count})' for val, count in top_values.items()])}")
                report.append("")
        
        return "\n".join(report)

    def _create_visualizations(self, df: pd.DataFrame) -> list:
        """Create and save visualizations"""
        
        plots_created = []
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Numeric variables - histograms
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            filename = f"histogram_{col.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(f"Histogram: {col}")
        
        # Categorical variables - bar charts
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.tight_layout()
            
            filename = f"barplot_{col.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(f"Bar chart: {col}")
        
        # Correlation heatmap for numeric variables
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            filename = f"correlation_heatmap_{timestamp}.png"
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append("Correlation heatmap")
        
        # Scatter plot for first two numeric variables
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            plt.figure(figsize=(10, 8))
            plt.scatter(df[col1], df[col2], alpha=0.6)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f'{col1} vs {col2}')
            plt.grid(True, alpha=0.3)
            
            filename = f"scatter_{col1.lower().replace(' ', '_')}_vs_{col2.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(f"Scatter plot: {col1} vs {col2}")
        
        return plots_created

    async def process_message(self, message: Message):
        """Process messages using simplified EDA workflow"""
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=message.body)],
            "user_input": message.body,
            "has_data": False,
            "dataframe": None,
            "eda_complete": False,
            "visualization_complete": False,
            "next_action": "process_data"
        }
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            
            final_state = None
            async for state in self.workflow.astream(initial_state, config):
                final_state = state
                print(f"EDA Step: {list(state.keys())[0]}")
            
            # Extract all messages from final state
            if final_state:
                latest_values = list(final_state.values())[-1]
                messages = latest_values.get("messages", [])
                
                # Combine all AI messages into response
                ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
                response_content = "\n\n".join(ai_messages)
                
                if not response_content:
                    response_content = "EDA and visualization workflow completed!"
            else:
                response_content = "EDA workflow failed to complete"
            
            # Add summary footer
            response_content += f"\n\n---\n📁 **Files saved to:** `{self.plots_dir}`\n🔄 **Provide new data for fresh analysis**"
            
            # Stream the response
            async def response_iterator():
                yield response_content
            
            await self.stream_message(response_iterator())
            
        except Exception as e:
            error_msg = f"**EDA Workflow Error:** {str(e)}"
            async def error_iterator():
                yield error_msg
            await self.stream_message(error_iterator())
