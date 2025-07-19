"""
Context Retrieval Nodes using PocketFlow Architecture

Specific node implementations for notebook analysis, knowledge search, and report generation.
"""

import logging
from typing import Dict, Any, Optional, List
from .pocketflow import Node

logger = logging.getLogger(__name__)


class NotebookAnalysisNode(Node):
    """Node that analyzes notebook content using existing tools."""
    
    def __init__(self, notebook_tools, **kwargs):
        super().__init__(**kwargs)
        self.notebook_tools = notebook_tools
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare notebook analysis."""
        user_query = shared.get("user_query", "")
        notebook_path = shared.get("notebook_path")
        
        # Extract notebook path from query if not provided
        if not notebook_path:
            notebook_path = self._extract_notebook_path(user_query)
        
        # Use default notebook for testing if none provided
        if not notebook_path:
            notebook_path = "/Users/jujonahj/jupyter-ai-personas/jupyter_ai_personas/data_science_persona/test_context_retrieval.ipynb"
        
        logger.info(f"📓 Analyzing notebook: {notebook_path}")
        
        return {
            "user_query": user_query,
            "notebook_path": notebook_path
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notebook analysis."""
        notebook_path = prep_res["notebook_path"]
        
        try:
            # Use existing notebook reader tool
            if self.notebook_tools and hasattr(self.notebook_tools[0], 'extract_rag_context'):
                context_result = self.notebook_tools[0].extract_rag_context(notebook_path)
                
                return {
                    "notebook_path": notebook_path,
                    "context_extracted": True,
                    "analysis_stage": "eda",  # Default for now
                    "libraries": ["pandas", "numpy", "matplotlib", "seaborn", "sklearn"],
                    "context_summary": context_result if isinstance(context_result, str) else "Notebook analyzed"
                }
            else:
                # Fallback analysis
                return {
                    "notebook_path": notebook_path,
                    "context_extracted": False,
                    "analysis_stage": "unknown",
                    "libraries": ["pandas", "numpy"],
                    "context_summary": "Basic analysis completed"
                }
        except Exception as e:
            logger.warning(f"Notebook analysis failed: {e}")
            return {
                "notebook_path": notebook_path,
                "context_extracted": False,
                "error": str(e),
                "context_summary": "Analysis failed, using defaults"
            }
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store notebook analysis results in shared state."""
        shared["notebook_analysis"] = exec_res
        return "default"
    
    def _extract_notebook_path(self, query: str) -> Optional[str]:
        """Extract notebook path from query."""
        if "notebook:" in query.lower():
            parts = query.split("notebook:")
            if len(parts) > 1:
                return parts[1].strip().split()[0]
        
        if ".ipynb" in query:
            words = query.split()
            for word in words:
                if word.endswith('.ipynb'):
                    return word
        
        return None


class KnowledgeSearchNode(Node):
    """Node that searches for relevant content using existing RAG tools."""
    
    def __init__(self, rag_tools, **kwargs):
        super().__init__(**kwargs)
        self.rag_tools = rag_tools
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare knowledge search with intelligent query generation."""
        user_query = shared.get("user_query", "")
        notebook_analysis = shared.get("notebook_analysis", {})
        libraries = notebook_analysis.get("libraries", ["pandas", "numpy"])
        context_summary = notebook_analysis.get("context_summary", "")
        
        logger.info(f"🔍 Preparing intelligent RAG search")
        
        # Generate contextual search queries based on notebook analysis
        contextual_queries = self._generate_contextual_queries(user_query, context_summary, libraries)
        
        return {
            "user_query": user_query,
            "libraries": libraries,
            "notebook_analysis": notebook_analysis,
            "contextual_queries": contextual_queries
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute intelligent RAG searches using contextual queries."""
        contextual_queries = prep_res["contextual_queries"]
        
        search_results = []
        
        if self.rag_tools and len(self.rag_tools) > 0:
            rag_tool = self.rag_tools[0]
            logger.info(f"🔍 RAG tool available: {type(rag_tool).__name__}")
            
            if hasattr(rag_tool, 'search_repository'):
                try:
                    logger.info(f"🧠 Executing {len(contextual_queries)} intelligent RAG searches")
                    
                    for i, query_info in enumerate(contextual_queries):
                        query = query_info["query"]
                        query_type = query_info["type"]
                        priority = query_info["priority"]
                        
                        logger.info(f"🔍 [{i+1}/{len(contextual_queries)}] {query_type} search (priority: {priority}): '{query}'")
                        
                        # Use higher k for high priority queries
                        k = 4 if priority == "high" else 3 if priority == "medium" else 2
                        
                        result = rag_tool.search_repository(query, k=k)
                        logger.info(f"📚 RAG results for '{query}':")
                        self._log_rag_results(result, "  ")
                        
                        search_results.append({
                            "query": query,
                            "type": query_type,
                            "priority": priority,
                            "result": result
                        })
                    
                    logger.info(f"✅ All intelligent RAG searches completed: {len(search_results)} total searches")
                    
                except Exception as e:
                    logger.error(f"❌ RAG search failed: {e}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    search_results.append({
                        "query": user_query,
                        "type": "error",
                        "error": str(e)
                    })
            else:
                logger.error(f"❌ RAG tool missing search_repository method: {dir(rag_tool)}")
                search_results.append({
                    "query": user_query,
                    "type": "error",
                    "error": "RAG tool missing search_repository method"
                })
        else:
            logger.error("❌ No RAG tools available")
            search_results.append({
                "query": user_query,
                "type": "error",
                "error": "No RAG tools available"
            })
        
        return search_results
    
    def _generate_contextual_queries(self, user_query: str, context_summary: str, libraries: List[str]) -> List[Dict[str, Any]]:
        """Generate intelligent, contextual search queries based on notebook analysis."""
        queries = []
        
        # Clean user query (remove file paths and persona mentions)
        clean_query = self._clean_user_query(user_query)
        
        # Extract key concepts from notebook context
        context_keywords = self._extract_context_keywords(context_summary)
        
        logger.info(f"🧠 Extracted context keywords: {context_keywords}")
        
        # 1. High Priority: Specific technical queries based on actual notebook content
        if context_keywords.get("techniques"):
            for technique in context_keywords["techniques"][:2]:  # Top 2 techniques
                queries.append({
                    "query": f"{technique} {' '.join(libraries[:2])} implementation examples",
                    "type": "technique_specific",
                    "priority": "high"
                })
        
        # 2. High Priority: Domain-specific queries
        if context_keywords.get("domain"):
            domain = context_keywords["domain"]
            primary_lib = libraries[0] if libraries else "python"
            queries.append({
                "query": f"{domain} analysis {primary_lib} workflow tutorial",
                "type": "domain_specific", 
                "priority": "high"
            })
        
        # 3. Medium Priority: Library-specific with context
        for lib in libraries[:2]:  # Top 2 libraries
            if context_keywords.get("operations"):
                operation = context_keywords["operations"][0]  # Top operation
                queries.append({
                    "query": f"{lib} {operation} advanced techniques examples",
                    "type": "library_contextual",
                    "priority": "medium"
                })
        
        # 4. Medium Priority: Problem-solving queries
        if context_keywords.get("problems"):
            problem = context_keywords["problems"][0]  # Top problem
            queries.append({
                "query": f"{problem} solution {' '.join(libraries[:2])} best practices",
                "type": "problem_solving",
                "priority": "medium"
            })
        
        # 5. Low Priority: Enhanced user query (only if specific and clean)
        if clean_query and len(clean_query.split()) > 2 and not any(x in clean_query.lower() for x in ["@", "ipynb", "/"]):
            queries.append({
                "query": f"{clean_query} {libraries[0] if libraries else 'python'} tutorial",
                "type": "user_query_enhanced",
                "priority": "low"
            })
        
        # Ensure we have at least a few queries
        if len(queries) < 3:
            # Add fallback queries
            queries.append({
                "query": f"{libraries[0] if libraries else 'pandas'} data analysis workflow examples",
                "type": "fallback",
                "priority": "medium"
            })
        
        logger.info(f"🎯 Generated {len(queries)} contextual queries")
        for i, q in enumerate(queries):
            logger.info(f"  [{i+1}] {q['priority'].upper()}: {q['query']}")
        
        return queries[:5]  # Limit to 5 queries max
    
    def _clean_user_query(self, query: str) -> str:
        """Clean user query by removing file paths and persona mentions."""
        import re
        
        # Remove file paths
        query = re.sub(r'/[^\s]*\.ipynb', '', query)
        # Remove persona mentions
        query = re.sub(r'@\w+', '', query)
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query.strip()
    
    def _extract_context_keywords(self, context_summary: str) -> Dict[str, List[str]]:
        """Extract meaningful keywords from notebook context."""
        keywords = {
            "techniques": [],
            "domain": None,
            "operations": [],
            "problems": []
        }
        
        context_lower = context_summary.lower()
        
        # Extract techniques/methods
        technique_patterns = [
            r"(linear regression|logistic regression|random forest|neural network|clustering|classification)",
            r"(cross validation|feature engineering|data preprocessing|model evaluation)",
            r"(visualization|plotting|analysis|prediction|forecasting)"
        ]
        
        for pattern in technique_patterns:
            import re
            matches = re.findall(pattern, context_lower)
            keywords["techniques"].extend(matches)
        
        # Extract domain
        domain_mapping = {
            "sales": ["sales", "revenue", "marketing", "advertising"],
            "finance": ["financial", "stock", "trading", "investment"],
            "healthcare": ["medical", "patient", "clinical", "health"],
            "business": ["business", "customer", "profit", "analytics"]
        }
        
        for domain, indicators in domain_mapping.items():
            if any(indicator in context_lower for indicator in indicators):
                keywords["domain"] = domain
                break
        
        # Extract operations
        operation_patterns = [
            r"(dataframe|data manipulation|data cleaning|feature selection)",
            r"(model training|model fitting|prediction|evaluation)",
            r"(plotting|visualization|charts|graphs)"
        ]
        
        for pattern in operation_patterns:
            import re
            matches = re.findall(pattern, context_lower)
            keywords["operations"].extend(matches)
        
        # Extract common problems/objectives
        if "predict" in context_lower or "forecast" in context_lower:
            keywords["problems"].append("prediction modeling")
        if "classify" in context_lower or "classification" in context_lower:
            keywords["problems"].append("classification")
        if "cluster" in context_lower:
            keywords["problems"].append("clustering analysis")
        if "visualiz" in context_lower or "plot" in context_lower:
            keywords["problems"].append("data visualization")
        
        return keywords
    
    def _log_rag_results(self, rag_result: str, indent: str = ""):
        """Log RAG search results in a readable format with quality filtering."""
        try:
            import json
            
            if isinstance(rag_result, str):
                result_data = json.loads(rag_result)
            else:
                result_data = rag_result
            
            if isinstance(result_data, dict) and "results" in result_data:
                query = result_data.get("query", "Unknown")
                total = result_data.get("total_results", 0)
                success = result_data.get("search_successful", False)
                
                # Filter results for quality
                filtered_results = self._filter_rag_results(result_data["results"])
                
                logger.info(f"{indent}📊 Query: '{query}' | Total: {total} | Quality results: {len(filtered_results)} | Success: {success}")
                
                for i, doc in enumerate(filtered_results[:3], 1):  # Show top 3 quality results
                    content = doc.get("content", "")[:150] + "..." if doc.get("content") else "No content"
                    notebook = doc.get("notebook_name", "Unknown")
                    source = doc.get("source", "Unknown")
                    cell_type = doc.get("cell_type", "Unknown")
                    quality_score = doc.get("quality_score", 0)
                    
                    logger.info(f"{indent}📄 [{i}] {notebook} ({cell_type}) - Quality: {quality_score:.2f}")
                    logger.info(f"{indent}    📍 Source: {source}")
                    logger.info(f"{indent}    📝 Content: {content}")
            else:
                logger.info(f"{indent}📋 Raw result: {str(rag_result)[:200]}...")
                
        except Exception as e:
            logger.warning(f"{indent}⚠️ Could not parse RAG result: {e}")
            logger.info(f"{indent}📋 Raw result: {str(rag_result)[:200]}...")
    
    def _filter_rag_results(self, results: List[Dict]) -> List[Dict]:
        """Filter RAG results to remove low-quality content."""
        filtered = []
        
        for result in results:
            content = result.get("content", "").strip()
            
            # Skip low-quality content
            if self._is_low_quality_content(content):
                continue
            
            # Add quality score
            quality_score = self._calculate_quality_score(content)
            result["quality_score"] = quality_score
            
            filtered.append(result)
        
        # Sort by quality score (descending)
        filtered.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        return filtered
    
    def _is_low_quality_content(self, content: str) -> bool:
        """Determine if content is low quality and should be filtered out."""
        if not content or len(content.strip()) < 20:
            return True
        
        content_lower = content.lower().strip()
        
        # Filter out pure titles/headers
        if content_lower.startswith('#') and len(content_lower.split('\n')) == 1:
            return True
        
        # Filter out just imports
        if content_lower.startswith(('import ', 'from ')) and len(content_lower.split('\n')) <= 2:
            return True
        
        # Filter out very short snippets
        if len(content.split()) < 5:
            return True
        
        # Filter out generic documentation stubs
        generic_phrases = [
            "for more information",
            "see the documentation",
            "refer to the guide",
            "check the manual"
        ]
        if any(phrase in content_lower for phrase in generic_phrases) and len(content.split()) < 20:
            return True
        
        return False
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for content (0-1, higher is better)."""
        if not content:
            return 0.0
        
        score = 0.0
        content_lower = content.lower()
        
        # Length factor (sweet spot around 100-500 chars)
        length = len(content)
        if 50 <= length <= 1000:
            score += 0.3
        elif length > 1000:
            score += 0.2
        
        # Code examples boost score
        if any(indicator in content for indicator in ['```', 'def ', 'import ', '= ', 'print(']):
            score += 0.3
        
        # Technical terms boost score
        technical_terms = [
            'dataframe', 'array', 'function', 'method', 'parameter',
            'example', 'tutorial', 'implementation', 'workflow'
        ]
        for term in technical_terms:
            if term in content_lower:
                score += 0.1
        
        # Penalize very generic content
        generic_terms = ['introduction', 'overview', 'basics', 'getting started']
        for term in generic_terms:
            if term in content_lower and len(content.split()) < 30:
                score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: List[Dict[str, Any]]) -> str:
        """Store search results in shared state."""
        shared["search_results"] = exec_res
        shared["total_searches"] = len(exec_res)
        return "default"


class ReportGenerationNode(Node):
    """Node that generates markdown reports using search results."""
    
    def __init__(self, file_tools, **kwargs):
        super().__init__(**kwargs)
        self.file_tools = file_tools
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare report generation."""
        user_query = shared.get("user_query", "")
        notebook_analysis = shared.get("notebook_analysis", {})
        search_results = shared.get("search_results", [])
        
        logger.info(f"📝 Generating markdown report")
        
        return {
            "user_query": user_query,
            "notebook_analysis": notebook_analysis,
            "search_results": search_results
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report."""
        user_query = prep_res["user_query"]
        search_results = prep_res["search_results"]
        
        # Log summary of sources used in report
        logger.info(f"📝 Generating report for query: '{user_query}'")
        logger.info(f"📚 Using {len(search_results)} RAG search results as sources")
        
        # Log source summary
        all_sources = set()
        for search in search_results:
            if "result" in search:
                try:
                    import json
                    result_data = json.loads(search["result"]) if isinstance(search["result"], str) else search["result"]
                    if isinstance(result_data, dict) and "results" in result_data:
                        for doc in result_data["results"]:
                            source = doc.get("notebook_name", "Unknown")
                            all_sources.add(source)
                except:
                    pass
        
        if all_sources:
            logger.info(f"📖 Report will include content from {len(all_sources)} handbook sources:")
            for source in sorted(all_sources):
                logger.info(f"    📄 {source}")
        
        return self._create_markdown_report(
            user_query,
            prep_res["notebook_analysis"],
            search_results
        )
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: str) -> str:
        """Save report and store in shared state."""
        # Save report using file tools
        try:
            if self.file_tools and hasattr(self.file_tools[0], 'save_file'):
                self.file_tools[0].save_file(contents=exec_res, file_name="repo_context.md")
            
            shared["final_report"] = exec_res
            shared["report_saved"] = True
            shared["report_filename"] = "repo_context.md"
            
            return "default"
        except Exception as e:
            logger.error(f"Report saving failed: {e}")
            shared["final_report"] = exec_res
            shared["report_saved"] = False
            shared["error"] = str(e)
            return "default"  # Continue even if save fails
    
    def _create_markdown_report(self, query: str, notebook_analysis: Dict, search_results: List) -> str:
        """Create a comprehensive markdown report similar to original persona."""
        
        libraries = notebook_analysis.get("libraries", [])
        notebook_path = notebook_analysis.get("notebook_path", "Not specified")
        context_summary = notebook_analysis.get("context_summary", "No analysis available")
        
        report = f"""# Context Retrieval Analysis Report

## Executive Summary
Analysis of your data science project with focus on: {query}

## Current Notebook Analysis
- **Notebook**: {notebook_path}
- **Libraries**: {', '.join(libraries)}
- **Analysis Stage**: {notebook_analysis.get('analysis_stage', 'Unknown')}

### Context Summary
{context_summary}

## Search Results Summary
Found {len(search_results)} relevant searches through the Python Data Science Handbook.

"""
        
        # Add search results if available
        if search_results:
            report += "## Relevant Resources\n\n"
            
            for i, result in enumerate(search_results[:5], 1):  # Limit to 5 results
                query_text = result.get("query", "Unknown")
                result_type = result.get("type", "general")
                
                report += f"**{i}. {result_type.title()} Search:** {query_text}\n\n"
                
                # Try to extract useful content from result
                if "result" in result:
                    try:
                        import json
                        result_data = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                        if isinstance(result_data, dict) and "results" in result_data:
                            docs = result_data["results"][:2]  # Top 2 results
                            for doc in docs:
                                content = doc.get("content", "")[:200] + "..." if doc.get("content") else "No content"
                                notebook_name = doc.get("notebook_name", "Unknown")
                                report += f"- **From {notebook_name}**: {content}\n\n"
                    except:
                        report += "- Content available in search results\n\n"
        
        report += """## Actionable Next Steps

1. **Immediate Actions**
   - Review the relevant examples from the handbook
   - Apply best practices to your current analysis
   - Optimize your code based on the recommendations

2. **Library-Specific Improvements**
"""
        
        for lib in libraries[:3]:
            report += f"   - Optimize {lib} usage based on handbook examples\n"
        
        report += """
3. **Best Practices**
   - Follow data science workflow patterns
   - Implement proper error handling
   - Document your methodology

## Summary
This report provides targeted recommendations based on your notebook analysis and the Python Data Science Handbook content.

Generated by Context Retrieval Persona using PocketFlow architecture.
"""
        
        return report