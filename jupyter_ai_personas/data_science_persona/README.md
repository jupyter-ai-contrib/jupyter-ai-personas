# 🤖 Advanced Data Science Persona

Data science agent that combines intelligent reasoning with automated machine learning capabilities. This persona uses PocketFlow architecture to analyze context, make decisions, and provide targeted data science assistance with AutoGluon integration.

## Key Features

### **Intelligent Decision Making**
- **AI-Powered Reasoning**: Uses LLM to analyze context and choose optimal actions
- **Multi-Domain Detection**: Automatically detects Tabular, Time-Series, and Multimodal data patterns
- **Context-Aware Routing**: Routes requests to specialized analysis nodes based on intent
- **Iterative Analysis**: Can perform multiple analysis rounds for complex problems

### **AutoML Integration**
- **Dataset-Specific Code Generation**: Creates customized AutoGluon code for your exact data structure
- **Multi-Domain Support**: Handles Tabular, Time-Series, and Multimodal machine learning
- **Configurable Training**: Adjustable time limits (120s for testing, 600s+ for production)
- **Model Leaderboards**: Automatic model ranking and performance comparison
- **Error-Resilient**: Robust error handling with fallback strategies

### **Smart Data Analysis**
- **Automatic Notebook Reading**: Intelligent parsing of Jupyter notebook content
- **DataFrame Extraction**: Extracts actual data structures from notebook outputs
- **Target Column Detection**: Smart inference of target variables for ML tasks
- **Domain Classification**: Automatic detection of problem type (classification/regression/forecasting)

### **Advanced Workflow**
- **Modular Architecture**: Clean separation between agent orchestration and specialized nodes
- **Context Persistence**: Maintains notebook context across multiple interactions
- **Dataset Recommendations**: Provides curated datasets when no data is available
- **Comprehensive Analysis**: Full project reviews with strategic recommendations

## Architecture Overview

```
DataScienceAgent (Flow Orchestrator)
├── DecideAction Node → AI-powered decision making
├── MLTraining Node → AutoGluon integration
├── DataAnalysis Node → Focused analysis
├── DataRecommendation Node → Dataset suggestions
├── CompleteAnalysis Node → Comprehensive reviews
└── GreetingNode → User onboarding
```

### **Core Components**

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **agent.py** | Main orchestrator and flow logic | Context loading, notebook analysis, decision routing |
| **nodes.py** | Specialized task handlers | ML training, data analysis, recommendations |
| **autogluon_tool.py** | AutoML engine | Dataset-specific code generation, model training |
| **dataset_recommendation_tool.py** | Data sourcing | Curated dataset recommendations by domain |
| **file_reader_tool.py** | Context extraction | Notebook parsing, content analysis |

## Use Cases & Examples

### **1. AutoML Model Training**
```python
# User: "Train a classification model on my sales data"
# Agent Process:
# 1. Analyzes notebook content → detects tabular data
# 2. Extracts DataFrame structure → finds target column  
# 3. Generates dataset-specific AutoGluon code
# 4. Provides leaderboard analysis code

# Generated Output:
"""
## 🤖 AutoGluon Tabular Solution (Dataset-Specific)

**Target:** customer_satisfaction
**Dataset Shape:** (1000, 8)
**Problem Type:** classification

```python
# AutoGluon Tabular ML Solution - Dataset Specific
from autogluon.tabular import TabularDataset, TabularPredictor

# Verify target column exists
if 'customer_satisfaction' not in sales_data.columns:
    print("⚠️ Target column 'customer_satisfaction' not found!")
    # Smart fallback logic...
else:
    actual_target = 'customer_satisfaction'

# Train AutoGluon model
predictor = TabularPredictor(
    label=actual_target,
    problem_type='classification',
    path='./autogluon_models/tabular_model'
).fit(
    TabularDataset(sales_data),
    time_limit=120,
    presets='best_quality'
)
```

## 🏆 View Model Leaderboard
```python
leaderboard = predictor.leaderboard()
print("🏆 AutoGluon Model Leaderboard:")
print(leaderboard.head(10))

best_model = leaderboard.iloc[0]['model']
best_score = leaderboard.iloc[0]['score_val']
print(f"\\n🥇 BEST MODEL: {best_model}")
print(f"📊 BEST SCORE: {best_score:.4f}")
```

### **2. Time Series Forecasting**
```python
# User: "Create a time series forecast for my daily sales"
# Agent Process:
# 1. Detects time-series patterns in notebook
# 2. Analyzes DataFrame structure for temporal data
# 3. Generates AutoGluon TimeSeriesPredictor code
# 4. Handles data formatting requirements automatically

# Generated Output:
"""
## 🔮 AutoGluon Time Series Solution (Dataset-Specific)

**Target:** daily_sales  
**Dataset Shape:** (365, 3)
**Forecast Length:** 7 steps

```python
# AutoGluon Time Series Forecasting Solution - Dataset Specific
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd

# Prepare time series data for AutoGluon
ts_data_formatted = sales_data.copy()

# Handle date index or column
if isinstance(ts_data_formatted.index, pd.DatetimeIndex):
    ts_data_formatted = ts_data_formatted.reset_index()
    timestamp_col = ts_data_formatted.columns[0]
else:
    # Auto-detect date columns
    date_cols = [col for col in ts_data_formatted.columns if 'date' in col.lower()]
    timestamp_col = date_cols[0] if date_cols else 'timestamp'

# Add required item_id column
ts_data_formatted['item_id'] = 'series_1'
ts_data_formatted = ts_data_formatted.rename(columns={timestamp_col: 'timestamp'})

# Train AutoGluon time series model
predictor = TimeSeriesPredictor(
    target='daily_sales',
    prediction_length=7,
    path='./autogluon_models/timeseries_model'
).fit(
    TimeSeriesDataFrame(ts_data_formatted),
    time_limit=120,
    presets='best_quality'
)
```

### **3. Multimodal Learning**
```python
# User: "Train a model on my product data with text and images"
# Agent Process:
# 1. Detects multimodal patterns (text + categorical + numerical)
# 2. Analyzes mixed data types in DataFrame
# 3. Generates AutoGluon MultiModalPredictor code
# 4. Handles automatic feature processing

# Generated Output:
"""
## AutoGluon Multimodal Solution (Dataset-Specific)

**Target:** customer_satisfaction
**Dataset Shape:** (2000, 9)

**Features:**
- Text: product_title, product_description
- Categorical: category, brand
- Numerical: price, rating, num_reviews

```python
# AutoGluon Multimodal ML Solution - Dataset Specific
from autogluon.multimodal import MultiModalPredictor

# Train AutoGluon multimodal model
predictor = MultiModalPredictor(
    label='customer_satisfaction',
    path='./autogluon_models/multimodal_model'
).fit(
    product_data,
    time_limit=240,  # Multimodal typically needs more time
    presets='best_quality'
)

print("✅ Multimodal training completed!")
print("🎯 Model handles text, images, and numerical data automatically!")
```

### **4. Dataset Recommendations**
```python
# User: "I need classification data for testing"
# Agent Process:
# 1. Detects no existing data in notebook
# 2. Routes to DataRecommendation Node
# 3. Provides curated dataset suggestions
# 4. Includes direct download links and descriptions

# Generated Output:
"""
## 📊 Curated Classification Datasets

**🎯 Recommended for: Classification Tasks**

### 1. **Iris Dataset** (Beginner-Friendly)
- **Size:** 150 samples, 4 features
- **Classes:** 3 flower species
- **Use Case:** Multi-class classification
- **Download:** `from sklearn.datasets import load_iris`

### 2. **Wine Quality Dataset** (Intermediate)
- **Size:** 6,497 samples, 11 features  
- **Classes:** Quality ratings (3-9)
- **Use Case:** Ordinal classification
- **Download:** UCI ML Repository

### 3. **Customer Churn Dataset** (Business)
- **Size:** 10,000 samples, 20 features
- **Classes:** Churn (Yes/No) 
- **Use Case:** Binary classification
- **Features:** Demographics, usage patterns, billing
"""
```

## ⚙️ Installation & Setup

### **Requirements**
```bash
# Core dependencies
pip install jupyter-ai pandas numpy scikit-learn

# AutoML capabilities
pip install autogluon

# AWS Bedrock integration (optional)
pip install boto3 agno

# Data visualization
pip install matplotlib seaborn
```

### **Configuration**
```python
# 1. Set up in Jupyter AI
{
  "model_provider": "bedrock",
  "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
}

# 2. Create repo_context.md
"""
# Project: Sales Forecasting
## Goals
- Predict daily sales revenue
- Identify seasonal patterns
- Optimize inventory management

## Current Status
- Historical data: 2 years
- Features: date, sales, promotions, weather
- Challenge: Handling seasonality and promotions
"""

# 3. Prepare your notebook with data
import pandas as pd
sales_data = pd.read_csv('sales.csv')
sales_data.head()  # Agent will detect this automatically
```

## 🔧 Advanced Configuration

### **Time Limit Customization**
```python
# Quick testing (default)
AutoGluonTool(default_time_limit=120)  # 2 minutes

# Production training  
AutoGluonTool(default_time_limit=600)  # 10 minutes

# Maximum quality
AutoGluonTool(default_time_limit=3600)  # 1 hour
```

### **Domain-Specific Settings**
```python
# The agent automatically scales time limits:
# - Tabular: default_time_limit
# - Multimodal: default_time_limit * 2  
# - Time-Series: default_time_limit * 2
```

## Performance & Capabilities

| Feature | Capability | Performance |
|---------|------------|-------------|
| **Decision Latency** | AI-powered routing | ~2-5 seconds |
| **Code Generation** | Dataset-specific | ~3-8 seconds |
| **Data Detection** | Auto-domain classification | >95% accuracy |
| **Notebook Size** | Content processing | Up to 2MB |
| **Model Training** | AutoGluon integration | 2min - 1hr |
| **Context Memory** | Session persistence | Full conversation |

## Test Notebooks Included

### **1. test_tabular.ipynb**
- **Purpose:** Standard tabular ML demonstration
- **Features:** Classification, regression examples
- **Data:** Synthetic customer data
- **Models:** RandomForest, XGBoost comparisons

### **2. test_time_series.ipynb** 
- **Purpose:** Time series forecasting
- **Features:** Trend, seasonality, forecasting
- **Data:** Synthetic daily sales data
- **Models:** ARIMA, AutoGluon TimeSeriesPredictor

### **3. test_multimodal.ipynb**
- **Purpose:** Mixed data type handling
- **Features:** Text + categorical + numerical
- **Data:** E-commerce product data
- **Models:** MultiModalPredictor demo

## Usage Patterns

### **Conversational Flow**
```
User: "Help me train a model on my data"
  ↓
Agent: Analyzes notebook → Detects tabular data → MLTrainingNode
  ↓
Output: Dataset-specific AutoGluon code + leaderboard analysis
```

### **Iterative Analysis**
```
User: "What's wrong with my model accuracy?"
  ↓
Agent: DecideAction → DataAnalysisNode → Focused debugging
  ↓
User: "How can I improve it?"
  ↓  
Agent: DecideAction → CompleteAnalysisNode → Strategic recommendations
```

### **Data Exploration**
```
User: "I need data for classification"
  ↓
Agent: Detects no data → DataRecommendationNode
  ↓
Output: Curated dataset suggestions with download links
```

## 🔍 Troubleshooting

### **Common Issues**

**"No data detected in notebook"**
```python
# Solutions:
1. Ensure DataFrame is displayed: df.head(), df.info()
2. Use explicit variable names: sales_data = pd.read_csv(...)
3. Run cells with data operations
4. Check notebook outputs are visible
```

**"AutoGluon not available"**
```python
# Install AutoGluon components:
pip install autogluon.tabular      # For tabular data
pip install autogluon.timeseries   # For time series  
pip install autogluon.multimodal   # For mixed data types
pip install autogluon              # Full installation
```

**"Time series requires item_id column"**
```python
# The agent automatically handles this:
# - Adds 'item_id' column for single time series
# - Formats timestamp columns correctly
# - Validates target column existence
```

### **Testing**
```bash
# Run test notebooks
jupyter notebook test_tabular.ipynb
jupyter notebook test_time_series.ipynb  
jupyter notebook test_multimodal.ipynb

# Test agent directly
python -c "from agent import DataScienceAgent; agent = DataScienceAgent()"
```

### **Code Style**
- Follow existing patterns in nodes.py
- Add comprehensive docstrings
- Include error handling and logging
- Test with various data types and sizes
