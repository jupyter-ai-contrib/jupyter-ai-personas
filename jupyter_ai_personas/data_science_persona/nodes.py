import logging
import yaml
from .pocketflow import Node
from .autogluon_tool import AutoGluonTool
from .dataset_recommendation_tool import DatasetRecommendationTool
from agno.models.message import Message as AgnoMessage
import pandas as pd
import re
from io import StringIO

logger = logging.getLogger(__name__)


class DecideAction(Node):
    """
    Decision-making node that analyzes the user query and context
    to determine the appropriate action for data science analysis.
    """
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare context for decision making"""
        return {
            "user_query": shared.get("user_query", ""),
            "repo_context": shared.get("repo_context", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "history": shared.get("history", ""),
            "previous_actions": shared.get("action_history", []),
            "has_data": shared.get("has_data", False),
            "primary_domain": shared.get("primary_domain", "unknown"),
            "data_summary": shared.get("data_analysis", {}).get("data_summary", "")
        }
    
    def exec(self, prep_res):
        """Use LLM to decide on the next action"""
        try:
            if not self.model_client:
                return self._default_action(prep_res)
            
            prompt = self._create_decision_prompt(prep_res)
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            if hasattr(response, 'content'):
                decision_text = response.content
            elif isinstance(response, dict):
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        decision_text = message_content[0].get('text', str(response))
                    else:
                        decision_text = str(message_content)
                else:
                    decision_text = str(response)
            else:
                decision_text = str(response)
            
            logger.debug(f"Raw LLM response: {decision_text[:200]}...")
            decision = self._parse_decision(decision_text)
            logger.info(f"🤖 Agent decided: {decision.get('action', 'unknown')}")
            return decision
            
        except Exception as e:
            logger.error(f"❌ Decision error: {e}")
            return self._default_action(prep_res)
    
    def _create_decision_prompt(self, prep_res):
        """Create prompt for decision making"""
        return f"""You are a data science agent analyzing a user request. Based on the context provided, decide what action to take next.

                USER QUERY: {prep_res['user_query']}

                DATA CONTEXT:
                - Has Data Available: {prep_res.get('has_data', False)}
                - Primary Domain: {prep_res.get('primary_domain', 'unknown')}
                - Data Summary: {prep_res.get('data_summary', 'No data summary available')}

                REPOSITORY CONTEXT:
                {prep_res['repo_context'][:1000] if prep_res['repo_context'] else 'No repo context available'}

                NOTEBOOK CONTENT:
                {prep_res['notebook_content'][:1500] if prep_res['notebook_content'] else 'No notebook content available'}

                NOTEBOOK PATH: {prep_res['notebook_path']}

                PREVIOUS ACTIONS: {prep_res['previous_actions']}

                Based on this context, decide what action to take. You MUST respond in valid YAML format.

                Choose ONE action from: analyze_data, generate_code, explain_concept, find_issues, create_visualization, debug_code, 
                train_ml_model, complete_analysis, greeting, recommend_datasets

                The action train_ml_model should be chosen only if the user specifically asks to train or fit a model, or if they are 
                asking to find the best model for the current stage.

                IMPORTANT: Respond with ONLY valid YAML. Do not include any other text.

                ```yaml
                action: [choose one from the list above]
                reasoning: [brief explanation in quotes]
                priority: [high, medium, or low]
                context_summary: [key points in quotes]
                next_steps: [what should happen after this action in quotes]
                ```

                Your YAML response:"""
                    
    def _parse_decision(self, decision_text):
        """Parse the LLM decision response with robust error handling"""
        try:
            yaml_content = decision_text.strip()
            
            # Try to find YAML block first
            if "```yaml" in decision_text:
                yaml_start = decision_text.find("```yaml") + 7
                yaml_end = decision_text.find("```", yaml_start)
                if yaml_end > yaml_start:
                    yaml_content = decision_text[yaml_start:yaml_end].strip()
            elif "```" in decision_text:
                yaml_start = decision_text.find("```") + 3
                yaml_end = decision_text.find("```", yaml_start)
                if yaml_end > yaml_start:
                    yaml_content = decision_text[yaml_start:yaml_end].strip()
            
            # Clean up common YAML issues
            yaml_content = self._clean_yaml_content(yaml_content)
            decision = yaml.safe_load(yaml_content)
            
            if not isinstance(decision, dict):
                logger.warning(f"Decision is not a dict: {type(decision)}")
                return self._extract_decision_from_text(decision_text)
            
            # Ensure required fields exist
            decision.setdefault("action", "complete_analysis")
            decision.setdefault("reasoning", "Fallback to complete analysis")
            decision.setdefault("priority", "medium")
            
            logger.debug(f"Parsed decision: {decision}")
            return decision
            
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            logger.debug(f"Raw YAML content: {yaml_content}")
            return self._extract_decision_from_text(decision_text)
        except Exception as e:
            logger.error(f"❌ Decision parsing error: {e}")
            return self._default_decision()
    
    def _clean_yaml_content(self, yaml_content):
        """Clean common YAML formatting issues"""
        yaml_content = yaml_content.strip()
        lines = yaml_content.split('\n')
        cleaned_lines = []
        for line in lines:
            if ':' in line and not line.strip().startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    cleaned_lines.append(f"{key}: {value}")
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_decision_from_text(self, text):
        """Extract decision from text when YAML parsing fails"""
        decision = self._default_decision()
        text_lower = text.lower()
        
        actions = ["analyze_data", "generate_code", "explain_concept", "find_issues", 
                  "create_visualization", "debug_code", "train_ml_model", 
                  "complete_analysis", "greeting", "recommend_datasets"]
        
        for action in actions:
            if action in text_lower:
                decision["action"] = action
                break
        
        # Extract reasoning by looking for common patterns
        if "reasoning" in text_lower or "because" in text_lower:
            for line in text.split('\n'):
                if any(word in line.lower() for word in ["reasoning", "because", "since"]):
                    decision["reasoning"] = line.strip()
                    break
        
        logger.warning(f"Used text extraction fallback: {decision}")
        return decision
    
    def _default_decision(self):
        """Default decision when LLM fails"""
        return {
            "action": "complete_analysis",
            "reasoning": "Fallback to complete analysis",
            "priority": "medium",
            "context_summary": "Limited context available",
            "next_steps": "Provide comprehensive analysis"
        }
    
    def _default_action(self, prep_res):
        """Default action when no model available - with simple data request detection"""
        user_query = prep_res.get("user_query", "").lower()
        has_data = prep_res.get("has_data", False)
        
        # Simple detection for data requests when no data is available
        data_request_indicators = [
            "need data", "want data", "give me data", "provide data", 
            "dataset", "classification data", "regression data", "training data",
            "sample data", "example data", "demo data"
        ]
        
        if not has_data and any(indicator in user_query for indicator in data_request_indicators):
            return {
                "action": "recommend_datasets",
                "reasoning": "User requesting data and none available",
                "priority": "high",
                "context_summary": "Data request detected, no data available",
                "next_steps": "Provide dataset recommendations"
            }
        
        return self._default_decision()
    
    def post(self, shared, prep_res, exec_res):
        """Update shared state with decision"""
        shared["current_action"] = exec_res.get("action", "complete_analysis")
        shared["action_reasoning"] = exec_res.get("reasoning", "")
        shared["action_priority"] = exec_res.get("priority", "medium")
        shared["context_summary"] = exec_res.get("context_summary", "")
        
        action_history = shared.get("action_history", [])
        action_history.append(exec_res.get("action", "complete_analysis"))
        shared["action_history"] = action_history
        action = exec_res.get("action", "complete_analysis")
        
        if action in ["analyze_data", "find_issues", "debug_code"]:
            return "analyze"
        elif action == "train_ml_model":
            return "ml_training"
        elif action == "recommend_datasets":
            return "recommend_data"
        elif action in ["generate_code", "create_visualization"]:
            return "complete"
        elif action == "explain_concept":
            return "complete"
        elif action == "greeting":
            return "greeting"
        else:
            return "complete"

class GreetingNode(Node):
    """Node for handling greetings and introductions"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for greeting"""
        return {
            "user_query": shared.get("user_query", ""),
            "history": shared.get("history", ""),
            "previous_actions": shared.get("action_history", [])
        }
    
    def exec(self, prep_res):
        """Execute greeting response"""
        try:
            # Check if this is a simple greeting
            query_lower = prep_res.get("user_query", "").lower()
            greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
            
            is_greeting = any(word in query_lower for word in greeting_words)
            
            if is_greeting and len(prep_res.get("user_query", "").split()) <= 5:
                greeting_response = """# Hello! 👋 Welcome to the Data Science Assistant
                
I'm your advanced data science agent, powered by sophisticated reasoning capabilities and ready to help you with:

## 🔬 **What I Can Do:**
- **Smart Data Analysis**: Analyze your datasets with targeted insights
- **Recommend Datasets**: Provide datasets based on specified requests
- **ML Model Training**: Automated machine learning with AutoGluon
- **Code Generation**: Ready-to-use Python code for your projects
- **Problem Solving**: Debug issues and optimize your analysis
- **Context-Aware Help**: I read your notebooks and project context automatically

## 🚀 **Getting Started:**
Just tell me what you'd like to work on! For example:
- "Analyze my sales data for trends"
- "Help me train a classification model" 
- "Debug my notebook: notebook_name.ipynb"
- "Optimize my data preprocessing pipeline"

I'll automatically read your repository context and notebook content to provide targeted, actionable recommendations.

What would you like to explore today? 🎯"""
                
                return {"greeting": greeting_response, "success": True}
            else:
                return {"greeting": "", "success": False, "route_to_analysis": True}
                
        except Exception as e:
            logger.error(f"❌ Greeting error: {e}")
            return {"greeting": "Hello! I'm ready to help with your data science tasks.", "success": True}
    
    def post(self, shared, prep_res, exec_res):
        """Handle greeting completion"""
        if exec_res.get("route_to_analysis"):
            return "complete"
        else:
            shared["final_response"] = exec_res.get("greeting", "Hello!")
            shared["analysis_complete"] = True
            return "end"

class DataAnalysisNode(Node):
    """Node for focused data analysis tasks"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for data analysis"""
        return {
            "user_query": shared.get("user_query", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_reasoning": shared.get("action_reasoning", "")
        }
    
    def exec(self, prep_res):
        """Execute focused data analysis"""
        try:
            if not self.model_client:
                return self._fallback_analysis(prep_res)
            
            prompt = f"""You are a data science expert performing focused data analysis.

                        USER REQUEST: {prep_res['user_query']}
                        CONTEXT: {prep_res['context_summary']}
                        REASONING: {prep_res['action_reasoning']}

                        NOTEBOOK CONTENT:
                        {prep_res['notebook_content'][:2000] if prep_res['notebook_content'] else 'No notebook content'}

                        Provide a focused analysis with:

                        ## 📊 Data Analysis
                        - Current data state and quality assessment
                        - Key patterns and insights from the data
                        - Statistical summary and observations

                        ## 🔍 Specific Findings
                        - Answer the user's specific question
                        - Highlight important data characteristics
                        - Identify potential issues or opportunities

                        ## 💡 Recommendations
                        - Specific next steps for this analysis
                        - Suggested improvements or additional analysis
                        - Priority actions based on findings

                        Focus on being specific and actionable rather than general."""
            
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            # Extract content from Bedrock response format
            if hasattr(response, 'content'):
                analysis = response.content
            elif isinstance(response, dict):
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        analysis = message_content[0].get('text', str(response))
                    else:
                        analysis = str(message_content)
                else:
                    analysis = str(response)
            else:
                analysis = str(response)
            
            return {"analysis": analysis, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return self._fallback_analysis(prep_res)
    
    def _fallback_analysis(self, prep_res):
        """Fallback when AI model unavailable for analysis"""
        return {
            "analysis": "## ❌ AI Model Unavailable\n\nData analysis requires AI model configuration. Please set up your AI model or try requesting dataset recommendations instead.",
            "success": False,
            "error": "AI model not configured"
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store analysis results"""
        shared["analysis_result"] = exec_res.get("analysis", "")
        shared["analysis_success"] = exec_res.get("success", False)
        return "decide"

class DataRecommendationNode(Node):
    """Dedicated node for providing dataset recommendations when no data is available"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
        self.dataset_tool = DatasetRecommendationTool()
    
    def prep(self, shared):
        """Prepare for dataset recommendation"""
        return {
            "user_query": shared.get("user_query", ""),
            "primary_domain": shared.get("primary_domain", "tabular"),
            "context_summary": shared.get("context_summary", ""),
            "action_reasoning": shared.get("action_reasoning", "")
        }
    
    def exec(self, prep_res):
        """Execute dataset recommendation"""
        try:
            logger.info("📊 Providing dataset recommendations - no data available")
            print("📊 DATA RECOMMENDATION: Providing curated datasets")
            
            result = self.dataset_tool.recommend_datasets(
                user_query=prep_res.get("user_query", ""),
                domain=prep_res.get("primary_domain", "Tabular"),
                max_results=5
            )
            
            if result.get("success"):
                logger.info("✅ Dataset recommendations generated successfully")
                return {
                    "success": True,
                    "recommendations": result.get("training_result", ""),
                    "domain": prep_res.get("primary_domain", "tabular")
                }
            else:
                logger.error("❌ Dataset recommendation failed")
                return {
                    "success": False,
                    "error": "Failed to generate dataset recommendations",
                    "recommendations": "Please provide your own dataset or check the dataset recommendation service."
                }
                
        except Exception as e:
            logger.error(f"❌ Dataset recommendation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": f"Dataset recommendation failed: {str(e)}"
            }
    
    def post(self, shared, prep_res, exec_res):
        """Store dataset recommendation results"""
        shared["final_response"] = exec_res.get("recommendations", "No recommendations available")
        shared["analysis_complete"] = True
        shared["recommendation_success"] = exec_res.get("success", False)
        return "end"


class MLTrainingNode(Node):
    """Node for automated machine learning training using AutoGluon - assumes data exists"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
        self.autogluon_tool = AutoGluonTool(default_time_limit=120)  # This impacts the number of models that can be trained, 120 for quick testing, 600 for optimal training
    
    def prep(self, shared):
        """Prepare for ML training using shared data analysis"""
        return {
            "user_query": shared.get("user_query", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_reasoning": shared.get("action_reasoning", ""),
            "data_analysis": shared.get("data_analysis", {}),
            "has_data": shared.get("has_data", False),
            "primary_domain": shared.get("primary_domain", "tabular")
        }
    
    def exec(self, prep_res):
        """Execute automated ML training - assumes data exists"""
        try:
            data_analysis = prep_res.get("data_analysis", {})
            has_data = prep_res.get("has_data", False)
            if not has_data:
                logger.error("❌ MLTrainingNode called without data - this is a routing error")
                return {
                    "success": False,
                    "error": "No data available for ML training. This request should have been routed to DataRecommendationNode.",
                    "training_result": "## ❌ ML Training Error\n\nNo data available. Please provide data or ask for dataset recommendations first."
                }
            
            training_type = prep_res.get("primary_domain", "tabular")
            logger.info(f"🎯 Selected AutoGluon domain: {training_type.upper()}")
            domain_mapping = {
                "Time-Series": "timeseries",
                "Multivariate": "multimodal", 
                "Tabular": "tabular"
            }
            autogluon_domain = domain_mapping.get(training_type, "tabular")
            logger.info(f"🤖 Using simplified AutoGluon tool for {training_type} domain")
            
            if data_analysis.get("success") and data_analysis.get("data_found"):
                logger.info("✅ Using agent's data analysis - generating dataset-specific AutoGluon code")                
                mock_notebook_data = {
                    "success": True,
                    "variable_name": data_analysis.get("variable_name", "df"),
                    "target_column": data_analysis.get("target_column"),
                    "problem_type": data_analysis.get("problem_type", "auto"),
                    "dataframe_info": {
                        "shape": data_analysis.get("characteristics", {}).get("shape", (100, 10)),
                        "columns": data_analysis.get("characteristics", {}).get("columns", []),
                        "dtypes": {}
                    }
                }
                
                try:
                    recommendation = self.autogluon_tool.generate_dataset_specific_code(
                        notebook_data=mock_notebook_data,
                        domain=autogluon_domain,
                        user_query=prep_res.get("user_query", "")
                    )
                    
                    if recommendation.get("success"):
                        leaderboard_section = ""
                        if recommendation.get("leaderboard_code"):
                            leaderboard_section = f"""

## 🏆 View Model Leaderboard

After training, run this code to see the best models:

```python
{recommendation.get('leaderboard_code', '')}
```"""
                        
                        result = {
                            "success": True,
                            "training_result": f"""{recommendation.get('solution_summary', '')}

```python
{recommendation.get('optimized_code', '')}
```{leaderboard_section}

*Generated specifically for your dataset structure - ready to run!*""",
                            "training_type": training_type,
                            "code_generated": True,
                            "dataset_specific": True
                        }
                    else:
                        logger.warning(f"⚠️ Dataset-specific code generation failed: {recommendation.get('error')}")
                        raise Exception(f"Dataset-specific generation failed: {recommendation.get('error')}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Dataset-specific code generation error: {e}")
                    pass
            
            if 'result' not in locals() or not result.get("success"):
                logger.error("❌ Dataset-specific code generation failed and generic code was removed")
                result = {
                    "success": False,
                    "error": "Dataset-specific code generation failed. Please ensure your notebook contains valid DataFrame data.",
                    "training_result": "## ❌ AutoGluon Error\n\nDataset-specific code generation failed. Generic templates have been removed for better accuracy. Please ensure your notebook contains properly formatted DataFrame data."
                }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ ML training error: {e}")
            return self._fallback_ml_training(prep_res)
    
    def _extract_data_from_notebook(self, notebook_content):
        """Extract actual DataFrames from notebook content string"""
        try:
            if not notebook_content:
                return {"success": False, "error": "No notebook content available"}
            
            dataframes = {}
            target_columns = []
            
            # Looks for DataFrame outputs (df.head(), df.info(), df.shape, etc.)
            cell_pattern = r"--- Cell \d+ \(CODE\) ---.*?SOURCE:\n(.*?)(?=OUTPUTS:|--- Cell|\Z)"
            output_pattern = r"OUTPUTS:\s*Output \d+ \([^)]+\):\s*(.*?)(?=\n\s*Output|\n--- Cell|\Z)"
            
            cells = re.findall(cell_pattern, notebook_content, re.DOTALL)
            
            for i, cell_source in enumerate(cells):
                df_assignments = re.findall(r"(\w+)\s*=.*?pd\.read_\w+\(", cell_source)
                display_commands = re.findall(r"(\w+)\.(?:head|tail|info|describe|shape|columns)", cell_source)
                
                # Combine variable names
                variable_names = list(set(df_assignments + display_commands))
                target_refs = re.findall(r"(?:y|target|label)\s*=\s*\w+\[['\"](.*?)['\"]\]", cell_source)
                target_columns.extend(target_refs)
            
            # Look for actual DataFrame output data in the outputs
            outputs = re.findall(output_pattern, notebook_content, re.DOTALL)
            
            for output in outputs:
                dataframe = self._parse_tabular_output(output.strip())
                if dataframe is not None:
                    var_name = variable_names[0] if variable_names else 'df'
                    dataframes[var_name] = dataframe
                    break 
            
            if dataframes:
                df_name, df = next(iter(dataframes.items()))
                
                target_col = None
                if target_columns:
                    for col in target_columns:
                        if col in df.columns:
                            target_col = col
                            break
                
                if not target_col:
                    target_col = self._infer_target_column_from_df(df)
                
                # Determine problem type
                problem_type = "classification"
                if target_col and target_col in df.columns:
                    if df[target_col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        unique_ratio = len(df[target_col].unique()) / len(df)
                        if unique_ratio > 0.1:  # More than 10% unique values suggests regression
                            problem_type = "regression"
                
                return {
                    "success": True,
                    "dataframe": df,
                    "target_column": target_col,
                    "problem_type": problem_type,
                    "variable_name": df_name,
                    "dataframe_info": {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict()
                    }
                }
            
            return {"success": False, "error": "No DataFrame data found in notebook outputs"}
            
        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_tabular_output(self, output_text):
        """Parse tabular output text to reconstruct DataFrame"""
        try:
            
            lines = output_text.strip().split('\n')
            
            # Pattern 1: Standard df.head() output with index and columns
            if any('  ' in line and not line.strip().startswith('[') for line in lines):
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('[') and not line.startswith('...'):
                        clean_lines.append(line)
                
                if len(clean_lines) >= 2:
                    try:
                        data_text = '\n'.join(clean_lines)
                        df = pd.read_csv(StringIO(data_text), sep=r'\s+', engine='python')
                        if len(df) > 0 and len(df.columns) > 1:
                            return df
                    except Exception:
                        pass
            
            # Pattern 2: CSV-like output
            if ',' in output_text and '\n' in output_text:
                try:
                    df = pd.read_csv(StringIO(output_text))
                    if len(df) > 0 and len(df.columns) > 1:
                        return df
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Tabular output parsing error: {e}")
            return None
    
    def _infer_target_column_from_df(self, df):
        """Infer likely target column from DataFrame structure"""
        target_names = ['target', 'label', 'y', 'class', 'category', 'outcome', 'result', 'price', 'value']
        
        for col in df.columns:
            if col.lower() in target_names:
                return col
        
        for col in df.columns:
            for target_name in target_names:
                if target_name in col.lower():
                    return col
        
        # Default: use last column (common ML convention)
        return df.columns[-1] if len(df.columns) > 0 else None
    
    def _fallback_ml_training(self, prep_res):
        """Fallback when AutoGluon unavailable"""
        return {
            "success": False,
            "training_result": "## ❌ AutoGluon Not Available\n\nML training requires AutoGluon. Install with: `pip install autogluon`\n\nAlternatively, ask for dataset recommendations to get started with data exploration.",
            "error": "AutoGluon not installed",
            "installation_required": True
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store ML training results"""
        shared["ml_training_result"] = exec_res.get("training_result", "")
        shared["ml_training_success"] = exec_res.get("success", False)
        shared["ml_model_path"] = exec_res.get("model_path", "")
        
        if exec_res.get("success"):
            # Set final response directly to ML training results - don't need complete analysis
            shared["final_response"] = exec_res.get("training_result", "")
            shared["analysis_complete"] = True
            return "end"
        else:
            return "decide"

class CompleteAnalysisNode(Node):
    """Node for comprehensive data science analysis"""
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
    
    def prep(self, shared):
        """Prepare for complete analysis"""
        return {
            "user_query": shared.get("user_query", ""),
            "repo_context": shared.get("repo_context", ""),
            "notebook_content": shared.get("notebook_content", ""),
            "notebook_path": shared.get("notebook_path", ""),
            "context_summary": shared.get("context_summary", ""),
            "action_history": shared.get("action_history", [])
        }
    
    def exec(self, prep_res):
        """Execute comprehensive analysis"""
        try:
            if not self.model_client:
                return self._fallback_complete_analysis(prep_res)
            
            prompt = f"""You are a senior data science expert providing comprehensive analysis and recommendations.

                    USER QUERY: {prep_res['user_query']}

                    REPOSITORY CONTEXT:
                    {prep_res['repo_context'][:1500] if prep_res['repo_context'] else 'No repo context available'}

                    NOTEBOOK CONTENT:
                    {prep_res['notebook_content'][:2500] if prep_res['notebook_content'] else 'No notebook content available'}

                    NOTEBOOK PATH: {prep_res['notebook_path']}

                    CONTEXT SUMMARY: {prep_res['context_summary']}

                    PREVIOUS ACTIONS: {prep_res['action_history']}

                    Provide a comprehensive data science analysis with:

                    ## 📊 Current State Analysis
                    - Thorough assessment of the current notebook content
                    - Data quality, structure, and completeness evaluation
                    - Current methodology and approach analysis
                    - Identification of strengths and weaknesses

                    ## 🎯 Targeted Recommendations
                    - Specific, actionable recommendations based on the user's query
                    - Priority-ordered suggestions for improvement
                    - Alternative approaches and methodologies to consider
                    - Best practices and optimization opportunities

                    ## 💻 Implementation Code
                    - Ready-to-use code snippets that can be directly implemented
                    - Proper imports and variable handling
                    - Comments explaining the approach and rationale
                    - Error handling and edge case considerations

                    ## 🔄 Next Steps Roadmap
                    - Clear, prioritized action items
                    - Timeline and dependency considerations
                    - Success metrics and validation approaches
                    - Long-term development suggestions

                    ## 🧪 Testing & Validation
                    - Suggested testing approaches for the analysis
                    - Validation methods for results
                    - Quality assurance recommendations
                    - Performance optimization suggestions

                    Focus on providing actionable, specific guidance that directly addresses the user's needs while building upon existing work."""
                                
            messages = [AgnoMessage(role="user", content=prompt)]
            response = self.model_client.invoke(messages)
            
            if hasattr(response, 'content'):
                complete_analysis = response.content
            elif isinstance(response, dict):
                if 'output' in response and 'message' in response['output']:
                    message_content = response['output']['message']['content']
                    if isinstance(message_content, list) and len(message_content) > 0:
                        complete_analysis = message_content[0].get('text', str(response))
                    else:
                        complete_analysis = str(message_content)
                else:
                    complete_analysis = str(response)
            else:
                complete_analysis = str(response)
            
            return {"complete_analysis": complete_analysis, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Complete analysis error: {e}")
            return self._fallback_complete_analysis(prep_res)
    
    def _fallback_complete_analysis(self, prep_res):
        """Fallback when AI model unavailable for complete analysis"""
        return {
            "complete_analysis": "## ❌ AI Model Unavailable\n\nComplete analysis requires AI model configuration. Please set up your AI model or ask for specific help like dataset recommendations.",
            "success": False,
            "error": "AI model not configured"
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store complete analysis results"""
        shared["final_response"] = exec_res.get("complete_analysis", "")
        shared["analysis_complete"] = True
        return "end"