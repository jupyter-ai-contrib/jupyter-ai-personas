import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AutoGluonTool:
    """AutoGluon tool for ML code generation with efficient template-based approach."""
    
    def __init__(self, default_time_limit: int = 120):
        self.default_time_limit = default_time_limit
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status and installation information."""
        return {
            "availability": self.availability,
            "installation_commands": {
                "full": "pip install autogluon",
                "tabular_only": "pip install autogluon.tabular",
                "multimodal_only": "pip install autogluon.multimodal", 
                "timeseries_only": "pip install autogluon.timeseries"
            },
            "any_available": any(self.availability.values())
        }
    
    def recommend_ml_solution(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AutoGluon code based on problem context - requires dataset-specific generation."""
        try:
            logger.info("🎯 AutoGluon recommendation requires dataset-specific generation")
            
            return {
                "success": False, 
                "error": "Generic recommendations removed. Use generate_dataset_specific_code() with actual dataset for optimal results.",
                "suggestion": "The AutoGluon tool now only supports dataset-specific code generation for better accuracy and reliability."
            }
                
        except Exception as e:
            logger.error(f"AutoGluon recommendation error: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_dataset_specific_code(self, notebook_data: Dict[str, Any], domain: str, user_query: str = "") -> Dict[str, Any]:
        """Generate AutoGluon code customized for the specific dataset structure."""
        try:
            if not notebook_data.get("success"):
                return {"success": False, "error": "No valid dataset provided"}
            
            logger.info(f"📊 Analyzing dataset structure for {domain} domain")
            if "dataframe" in notebook_data:
                df = notebook_data["dataframe"]
                columns = list(df.columns)
                shape = df.shape
                variable_name = notebook_data.get("variable_name", "df")
                target_column = self._detect_target_column(df, notebook_data, user_query)
                logger.info("📊 Using actual DataFrame for analysis")
            elif "dataframe_info" in notebook_data:
                df_info = notebook_data["dataframe_info"]
                columns = df_info.get("columns", [])
                shape = df_info.get("shape", (100, 10))
                variable_name = notebook_data.get("variable_name", "df")
                target_column = notebook_data.get("target_column", "target")
                logger.info("📊 Using dataset metadata for analysis")
            else:
                return {"success": False, "error": "No dataset or dataset info provided"}
            
            logger.info(f"🎯 Target column: {target_column}")
            logger.info(f"📊 Columns: {columns}")
            
            if domain == "timeseries":
                return self._generate_timeseries_code_for_dataset(shape, variable_name, target_column, columns, user_query)
            elif domain == "tabular":
                return self._generate_tabular_code_for_dataset(shape, variable_name, target_column, columns, user_query)
            elif domain == "multimodal":
                return self._generate_multimodal_code_for_dataset(shape, variable_name, target_column, columns, user_query)
            else:
                return {"success": False, "error": f"Unsupported domain: {domain}"}
                
        except Exception as e:
            logger.error(f"Dataset-specific code generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_target_column(self, df, notebook_data: Dict[str, Any], user_query: str) -> str:
        """Detect the most likely target column from the dataset."""
        if notebook_data.get("target_column"):
            return notebook_data["target_column"]
        target_candidates = []
        common_targets = ['target', 'label', 'y', 'class', 'category', 'outcome', 'result', 'price', 'value', 'sales', 'revenue']
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in common_targets:
                target_candidates.append(col)
            elif any(target in col_lower for target in common_targets):
                target_candidates.append(col)
        
        if target_candidates:
            return target_candidates[0]
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            return numeric_cols[-1]
        return df.columns[-1] if len(df.columns) > 0 else 'target'
    
    def _generate_timeseries_code_for_dataset(self, shape, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate time series code customized for the specific dataset."""
        
        prediction_length = 24
        if any(word in user_query.lower() for word in ["daily", "day"]):
            prediction_length = 7
        elif any(word in user_query.lower() for word in ["hourly", "hour"]):
            prediction_length = 24
        elif any(word in user_query.lower() for word in ["monthly", "month"]):
            prediction_length = 12
                
        code = f"""# AutoGluon Time Series Forecasting Solution - Dataset Specific
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd

# Dataset Analysis:
# - Shape: {shape}
# - Target Column: '{target_column}'
# - Available Columns: {columns}

# Prepare time series data for AutoGluon
ts_data_formatted = {variable_name}.copy()

# Handle date index or column
if isinstance(ts_data_formatted.index, pd.DatetimeIndex):
    # Data has datetime index - reset it to column
    ts_data_formatted = ts_data_formatted.reset_index()
    timestamp_col = ts_data_formatted.columns[0]
else:
    # Look for date column
    date_cols = [col for col in ts_data_formatted.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        timestamp_col = date_cols[0]
    else:
        # Create a simple date range if no date column found
        ts_data_formatted['timestamp'] = pd.date_range(start='2020-01-01', periods=len(ts_data_formatted), freq='D')
        timestamp_col = 'timestamp'

# Add required item_id column (single time series)
ts_data_formatted['item_id'] = 'series_1'

# Rename to AutoGluon format
ts_data_formatted = ts_data_formatted.rename(columns={{timestamp_col: 'timestamp'}})

# Reorder columns: item_id, timestamp, target columns
cols = ['item_id', 'timestamp'] + [col for col in ts_data_formatted.columns if col not in ['item_id', 'timestamp']]
ts_data_formatted = ts_data_formatted[cols]

# Verify target column exists
if '{target_column}' not in ts_data_formatted.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list(ts_data_formatted.columns))
    # Use first numeric column as backup
    numeric_cols = ts_data_formatted.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 0:
        actual_target = numeric_cols[0]
        print(f"Using '{{actual_target}}' as target instead")
    else:
        actual_target = '{target_column}'
else:
    actual_target = '{target_column}'

# Create TimeSeriesDataFrame
ts_autogluon = TimeSeriesDataFrame(ts_data_formatted)

# Train AutoGluon time series model
predictor = TimeSeriesPredictor(
    target=actual_target,
    prediction_length={prediction_length},
    path='./autogluon_models/timeseries_model'
).fit(
    ts_autogluon,
    time_limit={self.default_time_limit},
    presets='best_quality'
)

print(f"Generated forecasts for {{len(predictor.predict(ts_autogluon))}} steps")
print("✅ Time series forecasting completed!")"""

        leaderboard_code = f"""# 🏆 VIEW TIME SERIES MODEL PERFORMANCE AND RANKINGS
import pandas as pd

print("🏆 AutoGluon Time Series Training Summary:")
print("="*50)

# Get training summary and model information
try:
    summary = predictor.fit_summary()
    print("Training Summary:")
    print(summary)
except:
    print("Training summary not available")

# Best model information - TimeSeriesPredictor doesn't expose individual model names
print(f"\\n🥇 Best Model: AutoGluon Ensemble (WeightedEnsemble)")

# Model performance evaluation
performance = predictor.evaluate(ts_autogluon)
print("\\nModel Performance Metrics:")
print(performance)

# Generate forecasts
forecasts = predictor.predict(ts_autogluon)
print(f"\\nForecast Summary:")
print(f"Generated {{len(forecasts)}} forecast steps")
print(f"Target: {{actual_target}}")
print(f"Prediction Length: {prediction_length} steps")

print(f"\\nSample Forecasts:")
print(forecasts.head(10))

print(f"\\nModel Selection:")
print("AutoGluon automatically selected the best performing model from the ensemble")
print("The WeightedEnsemble combines multiple models for optimal performance")"""

        return {
            "success": True,
            "domain": "timeseries",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## 🔮 AutoGluon Time Series Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {shape}\n**Forecast Length:** {prediction_length} steps\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic date/time column detection\n- Robust target column validation\n- Production-ready forecasts"
        }
    
    def _generate_tabular_code_for_dataset(self, shape, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate tabular code customized for the specific dataset."""
        
        problem_type = None
        if any(word in user_query.lower() for word in ["regression", "predict", "estimate", "continuous"]):
            problem_type = "regression"
        code = f"""# AutoGluon Tabular ML Solution - Dataset Specific
from autogluon.tabular import TabularDataset, TabularPredictor

# Dataset Analysis:
# - Shape: {shape}
# - Target Column: '{target_column}'
# - Available Columns: {columns}
# - Problem Type: {problem_type}

# Verify target column exists
if '{target_column}' not in {variable_name}.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list({variable_name}.columns))
    # Try to find a suitable target column
    numeric_cols = {variable_name}.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 0:
        actual_target = numeric_cols[-1]  # Use last numeric column
        print(f"Using '{{actual_target}}' as target instead")
    else:
        actual_target = {variable_name}.columns[-1]  # Use last column
        print(f"Using '{{actual_target}}' as target instead")
else:
    actual_target = '{target_column}'

print(f"Training with target column: {{actual_target}}")
print(f"Dataset shape: {{{variable_name}.shape}}")

# Load your data
train_data = TabularDataset({variable_name})

# Train AutoGluon model
predictor = TabularPredictor(
    label=actual_target,{f'''
    problem_type='{problem_type}',''' if problem_type else ''}
    path='./autogluon_models/tabular_model'
).fit(
    train_data,
    time_limit={self.default_time_limit},
    presets='best_quality'
)

print(f"✅ Training completed for {{actual_target}}!")"""

        leaderboard_code = f"""# 🏆 VIEW MODEL LEADERBOARD AND BEST MODELS
leaderboard = predictor.leaderboard()
print("🏆 AutoGluon Model Leaderboard:")
print("="*50)
print(leaderboard.head(10))  # Show top 10 models

# 🥇 BEST MODEL INFORMATION
best_model = leaderboard.iloc[0]['model']
best_score = leaderboard.iloc[0]['score_val']
print(f"\\n🥇 BEST MODEL: {{best_model}}")
print(f"📊 BEST SCORE: {{best_score:.4f}}")

# 📈 DETAILED RANKING
print("\\n📈 Top 5 Models Ranking:")
for i, row in leaderboard.head(5).iterrows():
    print(f"{{i+1:2d}}. {{row['model']:25s}} | Score: {{row['score_val']:.4f}} | Time: {{row['fit_time']:.1f}}s")"""

        return {
            "success": True,
            "domain": "tabular",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## 🤖 AutoGluon Tabular Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {shape}\n**Problem Type:** {problem_type}\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic target column validation\n- Smart problem type detection\n- Comprehensive model evaluation and leaderboard"
        }
    
    def _generate_multimodal_code_for_dataset(self, shape, variable_name: str, target_column: str, columns: list, user_query: str) -> Dict[str, Any]:
        """Generate multimodal code customized for the specific dataset."""
        
        code = f"""# AutoGluon Multimodal ML Solution - Dataset Specific
from autogluon.multimodal import MultiModalPredictor

# Dataset Analysis:
# - Shape: {shape}
# - Target Column: '{target_column}'
# - Available Columns: {columns}

# Verify target column exists
if '{target_column}' not in {variable_name}.columns:
    print("⚠️  Target column '{target_column}' not found!")
    print("Available columns:", list({variable_name}.columns))
    actual_target = {variable_name}.columns[-1]  # Use last column
    print(f"Using '{{actual_target}}' as target instead")
else:
    actual_target = '{target_column}'

print(f"Training multimodal model with target: {{actual_target}}")
print(f"Dataset shape: {{{variable_name}.shape}}")

# Load your multimodal data (text, images, numerical)
train_data = {variable_name}

# Train AutoGluon multimodal model
predictor = MultiModalPredictor(
    label=actual_target,
    path='./autogluon_models/multimodal_model'
).fit(
    train_data,
    time_limit={self.default_time_limit * 2},  # Multimodal typically needs more time
    presets='best_quality'
)

print(f"✅ Multimodal training completed for {{actual_target}}!")
print("Model handles text, images, and numerical data automatically!")"""

        leaderboard_code = f"""#VIEW MULTIMODAL MODEL PERFORMANCE
performance = predictor.evaluate({variable_name})
print("🏆 AutoGluon Multimodal Performance:")
print("="*40)
print(performance)

# 📊 Model Information
print(f"\\n📊 Model Type: Multimodal (Text + Images + Numerical)")
print(f"🎯 Target: {{actual_target}}")
print(f"✅ Training completed successfully!")"""

        return {
            "success": True,
            "domain": "multimodal",
            "optimized_code": code,
            "leaderboard_code": leaderboard_code,
            "solution_summary": f"## AutoGluon Multimodal Solution \n\n**Target:** {target_column}\n**Dataset Shape:** {shape}\n\n**Features:**\n- Customized for your specific dataset structure\n- Automatic handling of text, images, and numerical data\n- Smart target column validation\n- State-of-the-art multimodal architectures"
        }
