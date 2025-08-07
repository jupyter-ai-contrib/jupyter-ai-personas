import logging
from pathlib import Path
import re
try:
    from .pocketflow import Flow
    from .file_reader_tool import NotebookReaderTool
    from .nodes import (
        DecideAction, GreetingNode, DataAnalysisNode, 
        DataRecommendationNode, MLTrainingNode, CompleteAnalysisNode
    )
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise ImportError(f"Missing dependencies for DataScienceAgent: {e}") from e

logger = logging.getLogger(__name__)

class DataScienceAgent(Flow):
    """ PocketFlow Agent that coordinates the workflow for Data Science Analysis """
    
    def __init__(self, model_client=None):
        super().__init__()
        self.model_client = model_client
        self._current_notebook_path = None
        self._current_notebook_content = None
        
        self.decide_node = DecideAction(model_client=model_client)
        self.greeting_node = GreetingNode(model_client=model_client)
        self.analyze_node = DataAnalysisNode(model_client=model_client)
        self.data_recommendation_node = DataRecommendationNode(model_client=model_client)
        self.ml_training_node = MLTrainingNode(model_client=model_client)
        self.complete_node = CompleteAnalysisNode(model_client=model_client)
        
        self.start(self.decide_node)
        
        self.decide_node - "greeting" >> self.greeting_node
        self.decide_node - "analyze" >> self.analyze_node
        self.decide_node - "recommend_data" >> self.data_recommendation_node
        self.decide_node - "ml_training" >> self.ml_training_node
        self.decide_node - "complete" >> self.complete_node
        
        self.analyze_node - "decide" >> self.decide_node
        self.greeting_node - "complete" >> self.complete_node
        self.ml_training_node - "decide" >> self.decide_node
        self.ml_training_node - "complete" >> self.complete_node

    def prep(self, shared):
        """Agent preparation - load context"""

        repo_context = self._load_repo_context()
        shared["repo_context"] = repo_context
        logger.info(f"📋 Repo context: {'✅ Loaded' if repo_context else '❌ Not found'}")
        user_query = shared.get("user_query", "")
        logger.debug(f"User query for notebook extraction: {user_query}")
        
        notebook_content, notebook_path, is_explicit = self._load_notebook_content(user_query)
        shared["notebook_content"] = notebook_content
        shared["notebook_path"] = notebook_path
        shared["notebook_explicit"] = is_explicit
        
        logger.info(f"📓 Notebook: {'✅ Loaded' if notebook_content else '❌ Not found'}")
        if notebook_path:
            logger.info(f"📁 Notebook path: {notebook_path} ({'explicit' if is_explicit else 'auto-discovered'})")
        
        logger.info("📊 Starting comprehensive data analysis...")
        data_analysis = self._analyze_all_available_data(user_query, notebook_content)
        
        shared["data_analysis"] = data_analysis
        shared["has_data"] = data_analysis.get("success", False)
        shared["data_characteristics"] = data_analysis.get("characteristics", {})
        shared["suggested_domains"] = data_analysis.get("suggested_domains", [])
        shared["primary_domain"] = data_analysis.get("primary_domain", "Tabular")
        
        if data_analysis.get("success"):
            chars = data_analysis.get("characteristics", {})
            logger.info(f"📋 Data shape: {chars.get('shape', 'unknown')}")
            logger.info(f"🎯 Suggested domain: {data_analysis.get('primary_domain', 'unknown')}")

        shared["action_history"] = []
        shared["analysis_complete"] = False
        
        prep_result = {
            "agent_initialized": True,
            "context_loaded": bool(repo_context),
            "notebook_loaded": bool(notebook_content)
        }
        
        logger.info(f"✅ Agent preparation complete: {prep_result}")
        return prep_result
    
    def _analyze_all_available_data(self, user_query, notebook_content):
        """Analyze existing notebook content for domain detection and code generation"""
        logger.info("📊 Analyzing notebook content for data characteristics...")        
        if not notebook_content:
            logger.warning("❌ No notebook content available for analysis")
            return {
                "success": False,
                "error": "No notebook content available",
                "primary_domain": "tabular"
            }
        
        try:
            analysis = self._analyze_notebook_data_characteristics(notebook_content)
            
            if analysis.get("success"):
                logger.info(f"✅ Data analysis successful: {analysis['primary_domain']} domain detected")
                return analysis
            else:
                logger.warning("⚠️ Data analysis completed but with limited information")
                return analysis
                
        except Exception as e:
            logger.error(f"❌ Data analysis error: {e}")
            print(f"❌ DATA TRACKER: Analysis ERROR - {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {e}",
                "primary_domain": "tabular"
            }
    
    def _analyze_notebook_data_characteristics(self, notebook_content):
        """Extract data characteristics from notebook content for domain detection"""
        try:
            analysis_result = {
                "success": False,
                "primary_domain": "tabular",
                "suggested_domains": ["tabular"],
                "data_found": False,
                "data_summary": "",
                "characteristics": {}
            }
            
            logger.info("🔍 Searching for DataFrame patterns in notebook...")
            print("🔍 DATA TRACKER: Searching for DataFrame patterns")
            
            dataframe_indicators = [
                r"(\w+)\s*=.*?pd\.read_\w+\(",  # df = pd.read_csv()
                r"(\w+)\s*=.*?DataFrame",        # df = DataFrame()
                r"(\w+)\.head\(\)",              # df.head()
                r"(\w+)\.info\(\)",              # df.info()
                r"(\w+)\.shape",                 # df.shape
                r"(\w+)\.describe\(\)"           # df.describe()
            ]
            
            found_variables = set()
            for pattern in dataframe_indicators:
                matches = re.findall(pattern, notebook_content, re.IGNORECASE)
                found_variables.update(matches)
            
            if found_variables:
                analysis_result["data_found"] = True
                analysis_result["success"] = True
                primary_var = list(found_variables)[0]
                analysis_result["variable_name"] = primary_var
                logger.info(f"📋 Found DataFrame variable: {primary_var}")
                print(f"📋 DATA TRACKER: Found DataFrame variable '{primary_var}'")
            
            shape_patterns = [
                r"\((\d+),\s*(\d+)\)",           # (1000, 5)
                r"(\d+)\s+rows?\s+×?\s*(\d+)\s+columns?",  # 1000 rows × 5 columns
                r"<class.*DataFrame.*>\[(\d+)\s+rows?\s+x\s+(\d+)\s+columns?\]"
            ]
            
            for pattern in shape_patterns:
                matches = re.findall(pattern, notebook_content, re.IGNORECASE)
                if matches:
                    rows, cols = matches[0]
                    analysis_result["characteristics"]["shape"] = (int(rows), int(cols))
                    analysis_result["data_summary"] = f"Shape: ({rows}, {cols})"
                    logger.info(f"📐 Found data shape: ({rows}, {cols})")
                    print(f"📐 SHAPE TRACKER: ({rows}, {cols})")
                    break
            
            column_patterns = [
                r"Index:\s*\[(.*?)\]",
                r"Columns:\s*\[(.*?)\]",
                r"columns=\[(.*?)\]",
                r"\.columns\s*=\s*\[(.*?)\]",
                r"columns:\s*\[(.*?)\]",
                r"Index\(.*?\[(.*?)\]",
            ]
            
            columns_found = []
            for pattern in column_patterns:
                matches = re.findall(pattern, notebook_content, re.IGNORECASE | re.DOTALL)
                if matches:
                    col_text = matches[0]
                    col_names = re.findall(r"['\"]([^'\"]+)['\"]", col_text)
                    if col_names:
                        columns_found = col_names[:10]
                        analysis_result["characteristics"]["columns"] = columns_found
                        logger.info(f"📋 Found columns: {columns_found}")
                        print(f"📋 COLUMNS TRACKER: {len(columns_found)} columns found")
                        break
            
            domain_scores = {"Tabular": 0, "Time-Series": 0, "Multivariate": 0}
            
            tabular_keywords = [
                r"classification", r"regression", r"predict", r"model\.fit",
                r"train_test_split", r"cross_validation", r"accuracy", r"precision",
                r"recall", r"sklearn", r"RandomForest", r"XGBoost", r"LogisticRegression"
            ]
            
            tabular_score = 10
            for keyword in tabular_keywords:
                if re.search(keyword, notebook_content, re.IGNORECASE):
                    tabular_score += 5
            
            domain_scores["Tabular"] = tabular_score
            logger.info(f"📊 Tabular indicators found (score: {tabular_score})")
            print(f"📊 TABULAR TRACKER: Score {tabular_score}")
            
            time_keywords = [
                r"pd\.to_datetime", r"datetime", r"timestamp", r"date", 
                r"time_series", r"forecast", r"trend", r"seasonal"
            ]
            
            time_score = 0
            for keyword in time_keywords:
                if re.search(keyword, notebook_content, re.IGNORECASE):
                    time_score += 10
            
            if time_score > 0:
                domain_scores["Time-Series"] = time_score
                logger.info(f"🕒 Time series indicators found (score: {time_score})")
                print(f"🕒 TIMESERIES TRACKER: Score {time_score}")
            
            multimodal_keywords = [
                r"text", r"image", r"nlp", r"cv2", r"PIL", 
                r"tokeniz", r"embedding", r"vision", r"language"
            ]
            
            multimodal_score = 0
            for keyword in multimodal_keywords:
                if re.search(keyword, notebook_content, re.IGNORECASE):
                    multimodal_score += 8
            
            if multimodal_score > 0:
                domain_scores["Multivariate"] = multimodal_score
                logger.info(f"🎭 Multimodal indicators found (score: {multimodal_score})")
                print(f"🎭 MULTIMODAL TRACKER: Score {multimodal_score}")
            
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            suggested_domains = [domain for domain, score in domain_scores.items() if score > 0]
            
            analysis_result.update({
                "primary_domain": primary_domain,
                "suggested_domains": suggested_domains,
                "domain_scores": domain_scores
            })
            
            target_patterns = [
                r"target\s*=\s*['\"]?(\w+)['\"]?",      # target = 'column_name'
                r"y\s*=\s*.*?\[?\s*['\"](\w+)['\"]",   # y = df['column_name']
                r"label\s*=\s*['\"]?(\w+)['\"]?",      # label = 'column_name' 
                r"predict\s*\(\s*['\"]?(\w+)['\"]?\s*\)"  # predict('column_name')
            ]
            
            for pattern in target_patterns:
                matches = re.findall(pattern, notebook_content, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], str) and matches[0]:
                        analysis_result["target_column"] = matches[0]
                        logger.info(f"🎯 Found target column: {matches[0]}")
                        print(f"🎯 TARGET TRACKER: Found '{matches[0]}'")
                    break
            
            if analysis_result["data_found"]:
                shape_info = analysis_result["characteristics"].get("shape", "unknown")
                col_count = len(columns_found) if columns_found else "unknown"
                analysis_result["data_summary"] = f"Shape: {shape_info}, Columns: {col_count}, Domain: {primary_domain}"
            
            logger.info(f"🎯 Domain analysis complete: {primary_domain}")
            print(f"🎯 FINAL DOMAIN: {primary_domain}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Notebook analysis error: {e}")
            print(f"❌ ANALYSIS ERROR: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {e}",
                "primary_domain": "tabular",
                "suggested_domains": ["tabular"]
            }
    def _load_repo_context(self):
        """Load repository context from repo_context.md"""
        try:
            repo_path = Path.cwd() / "repo_context.md"
            if repo_path.exists():
                with open(repo_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"❌ Error loading repo context: {e}")
        return ""
    
    def _load_notebook_content(self, user_query):
        """Load notebook content based on user query with persistence"""
        try:
            logger.debug(f"Loading notebook content for query: {user_query[:50]}...")
            
            notebook_info = self._extract_notebook_path(user_query)
            logger.debug(f"Extracted notebook info: {notebook_info}")
            
            if not notebook_info:
                if self._current_notebook_path and self._current_notebook_content:
                    logger.info(f"🔄 Using cached notebook: {self._current_notebook_path}")
                    return self._current_notebook_content, self._current_notebook_path, False
                else:
                    logger.warning("❌ No notebook path found and no cached content")
                    return "", "", False
            
            notebook_path = notebook_info["path"]
            is_explicit = notebook_info["explicit"]
            
            if str(notebook_path) != self._current_notebook_path:
                logger.info(f"📖 Loading {'explicit' if is_explicit else 'auto-discovered'} notebook: {notebook_path}")
                notebook_tool = NotebookReaderTool()
                content = notebook_tool.extract_rag_context(str(notebook_path))
                logger.debug(f"Notebook content length: {len(content)} characters")
                
                if content.startswith("Error:"):
                    logger.error(f"❌ Notebook reading failed: {content}")
                    return "", str(notebook_path), is_explicit
                else:
                    self._current_notebook_path = str(notebook_path)
                    self._current_notebook_content = content
                    logger.info(f"✅ Successfully loaded and cached notebook: {notebook_path}")
                    return content, str(notebook_path), is_explicit
            else:
                logger.info(f"🔄 Using cached notebook content for: {notebook_path}")
                return self._current_notebook_content or "", str(notebook_path), is_explicit
            
        except Exception as e:
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return "", "", False
    
    def _extract_notebook_path(self, query):
        """Extract notebook path from query"""
        import re
        
        ipynb_matches = re.findall(r'[\w\-_./\\]+\.ipynb', query)
        if not ipynb_matches:
            return None
        
        working_dir = Path.cwd()
        
        for match in ipynb_matches:
            notebook_path = Path(match)
            if not notebook_path.is_absolute():
                notebook_path = working_dir / notebook_path
            if notebook_path.exists():
                logger.info(f"✅ Found notebook: {notebook_path}")
                return {"path": notebook_path, "explicit": True}
        return None
    
    def run_analysis(self, user_query, **kwargs):
        """Run the data science agent analysis"""
        shared = {
            "user_query": user_query,
            "timestamp": kwargs.get("timestamp", ""),
            "history": kwargs.get("history", ""),
            **kwargs
        }
        
        logger.info(f"🤖 Starting agent analysis for: {user_query[:50]}...")
        logger.debug(f"Agent context: history={bool(kwargs.get('history'))}, timestamp={kwargs.get('timestamp')}")
        
        self.run(shared)
        
        logger.info(f"🤖 Agent analysis completed - Success: {shared.get('analysis_complete', False)}")
        logger.debug(f"Actions taken: {shared.get('action_history', [])}")
        
        return {
            "success": shared.get("analysis_complete", False),
            "response": shared.get("final_response", "No response generated"),
            "context_loaded": bool(shared.get("repo_context", "")),
            "notebook_loaded": bool(shared.get("notebook_content", "")),
            "notebook_path": shared.get("notebook_path", "") if shared.get("notebook_explicit", False) else "",
            "action_history": shared.get("action_history", []),
            "processing_summary": {
                "repo_context_loaded": bool(shared.get("repo_context", "")),
                "notebook_loaded": bool(shared.get("notebook_content", "")),
                "analysis_complete": shared.get("analysis_complete", False),
                "actions_taken": len(shared.get("action_history", []))
            }
        }
    
    def post(self, shared, prep_res, exec_res):
        """Agent completion"""
        shared["agent_completed"] = True
        logger.info(f"🤖 Agent completed - Actions taken: {len(shared.get('action_history', []))}")
        return exec_res