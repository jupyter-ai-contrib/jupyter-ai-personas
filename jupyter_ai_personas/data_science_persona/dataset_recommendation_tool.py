import logging
import re
import requests
import urllib.parse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TIMESERIES_KEYWORDS = ['time series', 'timeseries', 'temporal', 'forecast', 'forecasting', 'sequential', 'time-based', 
                      'stock', 'weather', 'climate', 'sales', 'financial', 'daily', 'hourly', 'monthly', 'yearly', 'seasonal']

DATA_TYPE_KEYWORDS = {
    'Time-Series': TIMESERIES_KEYWORDS,
    'Image': ['image', 'picture', 'visual', 'computer vision', 'photo', 'pixel', 'photograph', 'visual recognition'],
    'Text': ['text', 'document', 'nlp', 'natural language', 'corpus', 'linguistic', 'text analysis', 'document analysis'],
    'Multivariate': ['multivariate', 'multiple variables', 'multi-dimensional', 'several features', 'mixed data', 'multimodal'],
    'Sequential': ['sequence', 'sequential', 'ordered data', 'step by step', 'sequential analysis', 'ordered sequence'],
    'Spatio-Temporal': ['spatial', 'geographic', 'location', 'geo', 'coordinates']
}

TASK_KEYWORDS = {
    'Classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
    'Regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'predict value', 'forecast', 'forecasting'],
    'Clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
    'Recommendation-Systems': ['recommendation', 'recommender', 'collaborative filtering'],
    'Causal-Discovery': ['causal', 'causality', 'cause', 'effect', 'causal inference']
}

SUBJECT_KEYWORDS = {
    'Business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic', 'profit', 'revenue', 'stock', 'trading'],
    'Life-Sciences': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital', 'drug', 'cancer'],
    'Physical-Sciences': ['physics', 'chemistry', 'astronomy', 'energy', 'particle', 'chemical', 'molecular', 'weather', 'climate', 'temperature', 'sensor'],
    'CS-Engineering': ['computer', 'software', 'algorithm', 'network', 'system', 'engineering', 'technology', 'robot', 'ai', 'machine'],
    'Social-Sciences': ['social', 'psychology', 'sociology', 'demographic', 'census', 'population', 'survey', 'behavior', 'education', 'student'],
    'Game': ['game', 'chess', 'poker', 'tic-tac-toe', 'connect', 'puzzle', 'strategy'],
    'Law': ['legal', 'law', 'court', 'judge', 'crime', 'criminal', 'justice']
}

@dataclass
class Dataset:
    """Data class representing a dataset recommendation"""
    title: str
    description: str
    source: str  # "sample", "uci"
    domain: str  # UCI official: "Tabular", "Time-Series", "Sequential", "Multivariate", etc.
    url: str
    download_url: str
    size_mb: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    file_format: str = "csv"
    tags: List[str] = None
    difficulty: str = "beginner"  # "beginner", "intermediate", "advanced"
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        # Calculate size_mb if not provided
        if self.size_mb is None and self.rows and self.columns:
            self.size_mb = round(self.rows * self.columns * 0.001, 2)


class DatasetRecommendationTool:
    """Tool for finding and recommending datasets from online sources"""
    
    def __init__(self):
        """Initialize the dataset recommendation tool"""
        self.sources = {
            "uci": UCIMLRepoSource(),
        }
        logger.info("🔍 Dataset recommendation tool initialized")
    
    def recommend_datasets(self, user_query: str, domain: str = "Tabular", max_results: int = 5) -> Dict[str, Any]:
        """
        Main method to get dataset recommendations
        
        Args:
            user_query: User's original query/request
            domain: Detected UCI domain (Tabular, Time-Series, Sequential, Multivariate)
            max_results: Maximum number of datasets to recommend
            
        Returns:
            Dictionary with formatted results for agent
        """
        try:
            # Override domain based on query indicators
            detected_domain = self._detect_domain_from_query(user_query)
            if detected_domain:
                logger.info(f"🎯 Domain override: '{domain}' -> '{detected_domain}' based on query")
                print(f"🎯 DOMAIN OVERRIDE: {domain} -> {detected_domain}")
                domain = detected_domain
            
            logger.info(f"🔍 Searching for datasets: query='{user_query}', domain='{domain}'")
            print(f"🔍 DATASET SEARCH: Query='{user_query}', Domain='{domain}'")
            
            # Collect datasets from all sources
            all_datasets = []
            
            for source_name, source in self.sources.items():
                try:
                    logger.info(f"🔎 Searching {source_name} with semantic matching...")
                    datasets = source.search_datasets(
                        keywords=[user_query],
                        domain=domain,
                        max_results=max_results
                    )
                    all_datasets.extend(datasets)
                    logger.info(f"✅ Found {len(datasets)} datasets from {source_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Error searching {source_name}: {e}")
                    continue
            
            top_datasets = all_datasets[:max_results]
            logger.info(f"🎯 Returning {len(top_datasets)} dataset recommendations")
            print(f"🎯 DATASET RESULTS: {len(top_datasets)} recommendations found")
            return self._format_for_agent(top_datasets, domain)
            
        except Exception as e:
            logger.error(f"❌ Dataset recommendation error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Dataset Recommendation Error\n\n{str(e)}\n\nTry loading your own data with `pd.read_csv('your_data.csv')`"
            }
    
    def _format_for_agent(self, datasets: List[Dataset], domain: str) -> Dict[str, Any]:
        """Format dataset recommendations for agent consumption"""
        try:
            if not datasets:
                return {
                    "success": True,
                    "training_result": "## 📊 No Suitable Datasets Found\n\nNo datasets found matching your criteria. Try loading your own data with `pd.read_csv('your_data.csv')`"
                }
            
            displayed_count = min(len(datasets), 5)
            result_text = f"## 📊 Recommended Datasets\n\nBased on your query, here are {displayed_count} relevant datasets:\n\n"
            
            for i, dataset in enumerate(datasets[:5], 1):
                loading_code = self.generate_loading_code(dataset, "df")
                
                result_text += f"""### {i}. {dataset.title}
**Source:** {dataset.source.title()} | **Domain:** {dataset.domain.title()}
**Size:** {dataset.rows} rows × {dataset.columns} columns ({dataset.size_mb or 1.0}MB)

{dataset.description}

**Loading Code:**
```python
{loading_code}
```

---

"""
            
            result_text += "\n**Next Steps:** Choose a dataset above, run the loading code, then retry your ML training request."
            
            return {
                "success": True,
                "training_result": result_text
            }
            
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Formatting Error\n\n{str(e)}"
            }
    

    def generate_loading_code(self, dataset: Dataset, variable_name: str = "df") -> str:
        """Generate code to load a recommended dataset"""
        try:
            if dataset.source == "uci":
                return f"""
# Load {dataset.title} from UCI ML Repository
import pandas as pd

# Visit the UCI page to find the actual data file URL and replace below
# {variable_name} = pd.read_csv('paste_actual_download_url_here')
# print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
# {variable_name}.head()

print("UCI Dataset Page: {dataset.url}")
print("Visit the page above to find and download the dataset files")"""
            
            else:
                return f"""
# Load {dataset.title} dataset
import pandas as pd

{variable_name} = pd.read_csv('{dataset.download_url}')
print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
{variable_name}.head()"""
                
        except Exception as e:
            logger.warning(f"Code generation error: {e}")
            return f"# Error generating loading code for {dataset.title}"

    def _detect_domain_from_query(self, user_query: str) -> str:
        """Detect UCI domain from user query keywords"""
        query_lower = user_query.lower()
        
        # Check each data type for keyword matches
        for domain, keywords in DATA_TYPE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        # Default to None - don't override if no strong indicators
        return None


class UCIMLRepoSource:
    """Interface to UCI ML Repository with web scraping"""
    
    def __init__(self):
        self.base_url = "https://archive.ics.uci.edu"
        self.datasets_url = f"{self.base_url}/datasets"
        
    def search_datasets(self, keywords: List[str], domain: str, max_results: int = 5) -> List[Dataset]:
        """Search UCI ML Repository comprehensively to find best matching datasets"""
        try:
            
            logger.info("🔍 Starting comprehensive UCI database search...")
            
            user_query = ' '.join(keywords).lower()
            all_datasets = []
            seen_titles = set()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            search_strategies = self._generate_comprehensive_search_strategies(user_query, domain)
            
            target_dataset_count = max_results * 20  # Target 20x results for good selection
            
            for i, (strategy_name, base_params) in enumerate(search_strategies, 1):
                try:
                    logger.info(f"📋 Strategy {i}: {strategy_name}")
                    
                    strategy_datasets = 0
                    for page in range(10):
                        skip = page * 100
                        
                        # Build URL with pagination
                        if isinstance(base_params, str):
                            if '?' in base_params:
                                url = f"{base_params}&skip={skip}&take=100"
                            else:
                                url = f"{base_params}?skip={skip}&take=100&sort=desc&orderBy=NumHits"
                        else:  # It's parameters
                            data_type, task, subject = base_params
                            url = self._build_filter_url(data_type, task, subject, skip=skip, take=100)
                        
                        logger.debug(f"    Page {page+1}: {url}")
                        
                        response = requests.get(url, headers=headers, timeout=15)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.content, 'html.parser')
                        dataset_links = soup.find_all('a', href=lambda x: x and '/dataset/' in str(x))
                        logger.debug(f"    Found {len(dataset_links)} dataset links on page {page+1}")
                        
                        if not dataset_links:
                            break
                        
                        # Process all datasets found on this page
                        page_datasets = 0
                        for link in dataset_links:
                            dataset = self._parse_and_score_dataset(link, seen_titles, domain)
                            if dataset:
                                all_datasets.append(dataset)
                                seen_titles.add(dataset.title)
                                strategy_datasets += 1
                                page_datasets += 1
                        
                        logger.debug(f"    Processed {page_datasets} new datasets from page {page+1}")
                    
                    logger.info(f"    Total from strategy: {strategy_datasets} datasets")
                    
                    # Stop early if we have enough datasets from effective strategies  
                    if len(all_datasets) >= target_dataset_count:
                        logger.info(f"📊 Found {len(all_datasets)} datasets, stopping early to avoid inefficient strategies")
                        break
                        
                except Exception as e:
                    logger.debug(f"Strategy {i} ({strategy_name}) failed: {e}")
                    continue
            
            logger.info(f"📊 Total datasets collected: {len(all_datasets)}")
            
            if all_datasets:
                logger.info(f"📊 Filtering {len(all_datasets)} datasets using strict criteria...")
                keywords = user_query.split()
                filtered_datasets = []
                for dataset in all_datasets:
                    if self._matches_criteria(dataset, keywords, domain):
                        filtered_datasets.append(dataset)
                
                logger.info(f"✅ {len(filtered_datasets)} datasets passed criteria filter")
                
                if filtered_datasets:
                    final_datasets = filtered_datasets[:max_results]
                    logger.info(f"🏆 Returning top {len(final_datasets)} filtered datasets:")
                    for i, ds in enumerate(final_datasets, 1):
                        logger.info(f"  {i}. {ds.title}")
                    
                    return final_datasets
                else:
                    logger.warning("❌ No datasets passed the strict criteria filter")
                    return []
            else:
                logger.warning("❌ No datasets found in comprehensive search")
                return []
            
        except Exception as e:
            logger.error(f"❌ UCI comprehensive search error: {e}")
            print(f"❌ UCI COMPREHENSIVE SEARCH ERROR: {e}")
            return []
    
    def _generate_comprehensive_search_strategies(self, user_query: str, domain: str) -> List[tuple]:
        """Generate simplified search strategies (3 essential strategies)"""
        strategies = []
        
        primary_data_type = self._detect_uci_data_type(user_query, domain)
        primary_task = self._detect_uci_task(user_query)
        primary_subject = self._detect_uci_subject(user_query)
        
        # Strategy 1: Most specific search with detected filters
        if primary_data_type or primary_task or primary_subject:
            params = (primary_data_type, primary_task, primary_subject)
            strategies.append(("Targeted search", params))
        
        # Strategy 2: Domain-focused search
        params = (primary_data_type or "Tabular", "", "")
        strategies.append(("Domain search", params))
        
        # Strategy 3: Generic fallback
        params = ("Tabular", "Classification", "")
        strategies.append(("Fallback search", params))
        logger.info(f"🎯 Generated {len(strategies)} search strategies")
        return strategies
    
    def _parse_and_score_dataset(self, link, seen_titles: set, search_domain: str = None):
        """Parse dataset, avoiding duplicates (scoring done later after filtering)"""
        try:
            title_text = link.get_text().strip()
            if not title_text or len(title_text) < 2:
                return None
                
            # Skip duplicates
            if title_text in seen_titles:
                return None
            
            return self._parse_dataset_card(link, search_domain)
        except Exception as e:
            logger.debug(f"Error parsing dataset: {e}")
            return None
    
    def _detect_uci_data_type(self, user_query: str, domain: str) -> str:
        """Auto-detect data type filter from user query and domain"""
        query_lower = user_query.lower()
        
        for data_type, keywords in DATA_TYPE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return data_type
        
        # Domain-based defaults using UCI labels
        if domain == 'Time-Series':
            return 'Time-Series'
        elif domain in ['Multivariate', 'Image', 'Text']:
            return 'Multivariate'
        else:
            return 'Multivariate'
    
    def _detect_uci_task(self, user_query: str) -> str:
        """Auto-detect task filter from user query"""
        query_lower = user_query.lower()
        for task, keywords in TASK_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return task
        if any(kw in query_lower for kw in ['timeseries', 'time series', 'temporal']) and not any(kw in query_lower for kw in ['classification', 'regression', 'clustering']):
            return ''
        
        generic_terms = ['recommend', 'data', 'dataset']
        if any(term in query_lower for term in generic_terms) and len(query_lower.split()) <= 4:
            return ''
        return 'Classification'
    
    def _detect_uci_subject(self, user_query: str) -> str:
        """Auto-detect subject area filter from user query"""
        query_lower = user_query.lower()
        for subject, keywords in SUBJECT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return subject
        
        # For generic queries, return empty to avoid always getting same subject
        generic_terms = ['recommend', 'data', 'dataset', 'classification', 'regression', 'timeseries']
        if all(term in query_lower for term in generic_terms):
            return ''
        return ''
    
    def _build_filter_url(self, data_type: str, task: str, subject: str, skip: int = 0, take: int = 100) -> str:
        """Build UCI filter URL using correct API format"""
        
        base_url = "https://archive.ics.uci.edu/datasets"
        params = {
            'skip': skip,
            'take': take,
            'sort': 'desc',
            'orderBy': 'NumHits',
            'search': ''
        }
        
        if data_type:
            params['Types'] = data_type
        
        if task:
            params['Tasks'] = task
        
        if subject:
            params['Subjects'] = subject
        
        query_string = urllib.parse.urlencode(params)
        filter_url = f"{base_url}?{query_string}"
        
        return filter_url
    
    def _parse_dataset_card(self, card, expected_domain: str = None):
        """Parse a dataset card from UCI website"""
        try:
            name = ""
            dataset_url = ""
            description = ""
            
            # Method 1: Look for dataset link
            dataset_link = card.find('a', href=lambda x: x and '/dataset/' in str(x))
            if dataset_link:
                name = dataset_link.get_text().strip()
                href = dataset_link.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            # Method 2: If card IS the link
            elif card.name == 'a' and '/dataset/' in str(card.get('href', '')):
                name = card.get_text().strip()
                href = card.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            if not name or not dataset_url:
                return None
            
            # Extract description/abstract
            desc_element = (
                card.find('p') or 
                card.find('div', class_=lambda x: x and 'abstract' in str(x).lower()) or
                card.find('div', class_=lambda x: x and 'description' in str(x).lower())
            )
            if desc_element:
                description = desc_element.get_text().strip()
            
            instances = 100
            attributes = 10
            task_text = ""
            data_types_text = "Tabular"
            all_text = card.get_text().lower()
            
            instance_match = re.search(r'(\d+)\s*instances?', all_text)
            if instance_match:
                instances = int(instance_match.group(1))
                
            feature_match = re.search(r'(\d+)\s*features?', all_text)
            if feature_match:
                attributes = int(feature_match.group(1))
            
            if 'classification' in all_text:
                task_text = "Classification"
            elif 'regression' in all_text:
                task_text = "Regression"
            elif 'clustering' in all_text:
                task_text = "Clustering"
            else:
                task_text = "Classification"
            
            if expected_domain and expected_domain != "Tabular":
                domain = expected_domain
            else:
                domain = self._determine_domain(data_types_text, task_text, name)
            
            dataset = Dataset(
                title=name,
                description=description or f"UCI ML Repository dataset for {task_text.lower()}",
                source="uci",
                domain=domain,
                url=dataset_url,
                download_url=self._construct_download_url(dataset_url),
                rows=instances,
                columns=attributes,
                tags=self._extract_tags(task_text, data_types_text, name)
            )
            
            logger.debug(f"📊 Parsed UCI dataset: {name}")
            return dataset
            
        except Exception as e:
            logger.debug(f"Card parsing error: {e}")
            return None
    
    def _determine_domain(self, data_types: str, task: str, name: str):
        """Use UCI's official domain from data_types, with enhanced fallback detection"""

        if data_types and data_types != "Tabular":
            uci_type = data_types.strip()
            if uci_type in ['Time-Series', 'Sequential', 'Multivariate', 'Univariate', 'Text', 'Image', 'Other']:
                return uci_type
        text_lower = f"{data_types} {task} {name}".lower()
        
        if any(indicator in text_lower for indicator in ['time series', 'timeseries', 'temporal', 'forecast', 'stock', 'weather', 'climate', 'sales', 'financial', 'daily', 'hourly', 'monthly', 'yearly', 'seasonal']):
            return "Time-Series"
        elif any(indicator in text_lower for indicator in ['image', 'vision', 'photo', 'picture', 'pixel', 'visual']):
            return "Image"
        elif any(indicator in text_lower for indicator in ['text', 'nlp', 'language', 'speech', 'document', 'corpus']):
            return "Text"
        elif any(indicator in text_lower for indicator in ['sequential', 'sequence', 'ordered']):
            return "Sequential"
        elif any(indicator in text_lower for indicator in ['multivariate', 'mixed', 'multiple variables']):
            return "Multivariate"
        return "Tabular"
    
    
    def _extract_tags(self, task: str, data_types: str, name: str):
        """Extract basic tags from dataset info"""
        tags = []
        text_lower = f"{task} {data_types} {name}".lower()
        
        if 'classification' in text_lower:
            tags.append('classification')
        if 'regression' in text_lower:
            tags.append('regression')
        if any(kw in text_lower for kw in ['time', 'temporal', 'series']):
            tags.append('time-series')
        if any(kw in text_lower for kw in ['medical', 'health', 'clinical']):
            tags.append('medical')
        if any(kw in text_lower for kw in ['business', 'finance', 'economic']):
            tags.append('business')
            
        return tags[:3]
    
    def _construct_download_url(self, dataset_url: str):
        """Use the actual dataset URL as download link"""
        return dataset_url
    
    def _matches_criteria(self, dataset: Dataset, keywords: List[str], domain: str):
        """Check if dataset matches search criteria"""
        if not keywords:
            return True
        if domain != "Tabular" and dataset.domain != domain:
            return False
            
        searchable_text = f"{dataset.title} {dataset.description} {' '.join(dataset.tags)}".lower()
        query_text = ' '.join(keywords).lower()
        important_matches = False
        
        for category_dict in [TASK_KEYWORDS, SUBJECT_KEYWORDS, DATA_TYPE_KEYWORDS]:
            for category, keywords_list in category_dict.items():
                if any(kw in query_text for kw in keywords_list):
                    if any(kw in searchable_text for kw in keywords_list) or category.lower() in searchable_text:
                        important_matches = True
                        break
            if important_matches:
                break
        
        keyword_matches = 0
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() in searchable_text:
                keyword_matches += 1
        
        if important_matches:
            return True
        elif keyword_matches >= 1:
            return True
        elif domain in ["Tabular", "Multivariate"] and keyword_matches > 0:
            return True
        else:
            generic_terms = ['recommend', 'dataset', 'data']
            if any(term in ' '.join(keywords) for term in generic_terms):
                return True
            return False