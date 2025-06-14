import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from newsapi import NewsApiClient
import aiohttp
from bs4 import BeautifulSoup
import json
from pathlib import Path
import asyncio
import re
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsAPIHandler:
    def __init__(self):
        """Initialize the NewsAPI handler."""
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = Config.NEWS_API_KEY
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        
        # Initialize NewsAPI client
        self.client = NewsApiClient(api_key=self.api_key)
        
        # Create cache directory with absolute path
        self.cache_dir = Path(__file__).parent.parent / "data" / "news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Single source of truth for category mapping
        self.category_mapping = {
            "Economy": {
                "api": "business",
                "rag": "economy",
                "keywords": [
                    'economy', 'business', 'market', 'finance', 'trade', 'stock', 'investment',
                    'economic', 'financial', 'banking', 'currency', 'dollar', 'euro', 'yen',
                    'inflation', 'recession', 'growth', 'gdp', 'employment', 'unemployment',
                    'industry', 'corporate', 'company', 'enterprise', 'startup', 'venture',
                    'profit', 'revenue', 'income', 'wealth', 'tax', 'budget', 'debt'
                ]
            },
            "Science": {
                "api": "science",
                "rag": "science",
                "keywords": [
                    'science', 'research', 'study', 'discovery', 'scientist', 'experiment',
                    'laboratory', 'lab', 'physics', 'chemistry', 'biology', 'astronomy',
                    'space', 'universe', 'planet', 'star', 'galaxy', 'moon', 'mars',
                    'genetics', 'dna', 'evolution', 'species', 'ecosystem', 'climate',
                    'environment', 'technology', 'innovation', 'breakthrough', 'theory',
                    'hypothesis', 'analysis', 'data', 'evidence', 'observation'
                ]
            },
            "Technology": {
                "api": "technology",
                "rag": "technology",
                "keywords": [
                    'technology', 'tech', 'digital', 'software', 'hardware', 'computer', 'internet',
                    'artificial intelligence', 'ai', 'machine learning', 'ml', 'data science',
                    'programming', 'coding', 'developer', 'app', 'application', 'mobile',
                    'smartphone', 'laptop', 'tablet', 'device', 'gadget', 'innovation',
                    'startup', 'cybersecurity', 'privacy', 'blockchain', 'crypto',
                    'virtual reality', 'vr', 'augmented reality', 'ar', 'robotics',
                    'automation', 'cloud', 'server', 'network', 'wireless', '5g'
                ]
            },
            "Health": {
                "api": "health",
                "rag": "health",
                "keywords": [
                    'health', 'medical', 'disease', 'treatment', 'doctor', 'hospital', 'medicine',
                    'healthcare', 'patient', 'clinic', 'pharmacy', 'drug', 'vaccine',
                    'virus', 'infection', 'symptom', 'diagnosis', 'therapy', 'surgery',
                    'mental health', 'psychology', 'wellness', 'fitness', 'nutrition',
                    'diet', 'exercise', 'lifestyle', 'prevention', 'research', 'study',
                    'clinical trial', 'emergency', 'ambulance', 'paramedic', 'nurse',
                    'specialist', 'pediatric', 'geriatric', 'oncology', 'cardiology'
                ]
            },
            "Environment": {
                "api": "general",
                "rag": "environment",
                "keywords": [
                    'environment', 'climate', 'nature', 'pollution', 'conservation', 'earth', 'planet',
                    'global warming', 'climate change', 'greenhouse', 'emission', 'carbon',
                    'renewable', 'solar', 'wind', 'energy', 'sustainable', 'sustainability',
                    'ecosystem', 'biodiversity', 'wildlife', 'animal', 'plant', 'forest',
                    'ocean', 'sea', 'river', 'mountain', 'desert', 'wildlife', 'habitat',
                    'recycling', 'waste', 'plastic', 'pollution', 'clean', 'green',
                    'conservation', 'preservation', 'protection', 'natural resource'
                ]
            }
        }
        
        # Frontend categories
        self.frontend_categories = list(self.category_mapping.keys())
        
        # Add RAG document preparation
        self.rag_documents_dir = Path(__file__).parent.parent / "data" / "rag_documents"
        self.rag_documents_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_category(self, category: str) -> str:
        """Normalize category name to match backend expectations."""
        if category.lower() == "all":
            return "all"
        # Capitalize first letter of each word
        normalized = " ".join(word.capitalize() for word in category.split())
        # Validate against known categories
        if normalized not in self.category_mapping and normalized != "all":
            logger.warning(f"Unknown category: {normalized}, defaulting to 'Science'")
            return "Science"
        return normalized
    
    async def fetch_articles(
        self,
        category: str = "general",
        days: int = 1,
        max_articles: int = 10,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI with improved error handling."""
        try:
            # Validate category
            if category not in self.category_mapping:
                logger.error(f"Invalid category: {category}")
                return []  # Return empty list instead of raising error
            
            # Map the category to NewsAPI category
            api_category = self.category_mapping[category]["api"]
            logger.info(f"Mapping category '{category}' to NewsAPI category '{api_category}'")
            
            # Fetch articles
            logger.info(f"Fetching {api_category} articles from NewsAPI")
            try:
                response = self.client.get_top_headlines(
                    category=api_category,
                    language=language,
                    page_size=max_articles
                )
            except Exception as e:
                logger.error(f"NewsAPI request failed: {str(e)}")
                return []  # Return empty list on API error
            
            if response['status'] != 'ok':
                logger.error(f"NewsAPI error: {response.get('message', 'Unknown error')}")
                return []
            
            if not response.get('articles'):
                logger.warning(f"No articles found for category: {category}")
                return []
            
            # Process articles and prepare for RAG
            articles = []
            for article in response['articles']:
                try:
                    processed = await self._process_article(article)
                    if processed and self._validate_article_content(processed, category):
                        # Prepare for RAG
                        rag_doc = self.prepare_article_for_rag(processed)
                        if rag_doc:
                            articles.append(processed)
                except Exception as e:
                    logger.error(f"Error processing article {article.get('title', '')}: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles for category {category}: {str(e)}")
            return []  # Return empty list instead of raising error
    
    async def _process_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single article."""
        try:
            logger.info(f"Processing article: {article.get('title', '')}")
            
            # Extract main content
            content = await self._extract_main_content(article['url'])
            if not content:
                logger.warning(f"Could not extract content for article: {article.get('title', '')}")
                return None
            
            # Use description as fallback if content is too short
            if len(content) < 200 and article.get('description'):
                content = article['description']
                logger.info("Using article description as fallback content")
            
            logger.info(f"Successfully extracted content, length: {len(content)}")
            
            # Categorize article
            category = self._categorize_article(content)
            logger.info(f"Categorized article as: {category}")
            
            # Generate a unique ID for the article
            article_id = f"{category}_{article['publishedAt'].replace(':', '-').replace('+', '-')}"
            
            processed_article = {
                'id': article_id,  # Add ID field
                'title': article['title'],
                'source': article['source']['name'],
                'url': article['url'],
                'published_at': article['publishedAt'],
                'category': category,
                'content': content
            }
            
            logger.info(f"Successfully processed article: {article.get('title', '')}")
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return None
    
    async def _extract_main_content(self, url: str) -> Optional[str]:
        """Extract main content from article URL with timeout and retries."""
        # Skip known problematic domains
        problematic_domains = ['investors.micron.com', 'barrons.com', 'marketwatch.com']
        if any(domain in url for domain in problematic_domains):
            logger.warning(f"Skipping problematic domain: {url}")
            return None

        # Browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=5)  # Reduced timeout to 5 seconds
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                html = await response.text()
                                logger.info(f"Successfully fetched HTML, length: {len(html)}")
                                
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # Remove unwanted elements
                                for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                                    element.decompose()
                                
                                # Try to find the main article content
                                article_content = None
                                
                                # Try common article content containers
                                content_divs = soup.find_all(['article', 'div'], class_=re.compile(r'article|content|post|story'))
                                
                                if content_divs:
                                    logger.info(f"Found {len(content_divs)} potential content divs")
                                    # Get the largest content div
                                    content_div = max(content_divs, key=lambda x: len(x.get_text()))
                                    article_content = content_div.get_text(separator='\n', strip=True)
                                    logger.info(f"Extracted content from div, length: {len(article_content)}")
                                else:
                                    logger.warning("No content divs found, falling back to paragraph text")
                                    # Fallback to all paragraphs
                                    paragraphs = soup.find_all('p')
                                    if paragraphs:
                                        article_content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
                                        logger.info(f"Extracted content from paragraphs, length: {len(article_content)}")
                                    else:
                                        logger.warning("No paragraphs found, falling back to body text")
                                        article_content = soup.body.get_text(separator='\n', strip=True)
                                        logger.info(f"Extracted content from body, length: {len(article_content)}")
                                
                                if not article_content:
                                    logger.error("No content could be extracted")
                                    return "Article content could not be extracted. Please visit the source website for the full article."
                                
                                # Clean up the content
                                article_content = re.sub(r'\n\s*\n', '\n\n', article_content)  # Remove extra newlines
                                article_content = re.sub(r'\[.*?\]', '', article_content)  # Remove [text] patterns
                                article_content = re.sub(r'\+.*?chars', '', article_content)  # Remove [+N chars]
                                
                                logger.info(f"Final cleaned content length: {len(article_content)}")
                                return article_content.strip()
                            else:
                                logger.error(f"Failed to fetch URL, status code: {response.status}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(1)  # Wait before retry
                                    continue
                                return None
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout while fetching URL: {url}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    except Exception as e:
                        logger.error(f"Error fetching URL: {str(e)}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
            except Exception as e:
                logger.error(f"Error in _extract_main_content: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
        return None
    
    def _validate_article_content(self, article: Dict[str, Any], category: str) -> bool:
        """Validate article content for specific category requirements."""
        try:
            if not article.get('content'):
                logger.warning(f"No content found for article: {article.get('title', '')}")
                return False
                
            content = article['content'].lower()
            keywords = self.category_mapping[category]["keywords"]
            
            # Log the content length and first 100 characters
            logger.info(f"Content length: {len(content)}")
            logger.info(f"Content preview: {content[:100]}")
            
            # Check if content contains category-specific keywords
            keyword_matches = []
            for keyword in keywords:
                if keyword in content:
                    keyword_matches.append(keyword)
            
            # Log found keywords
            logger.info(f"Found keywords for {category}: {keyword_matches}")
            
            # Require at least 1 keyword match for validation (reduced from 2)
            is_valid = len(keyword_matches) >= 1
            logger.info(f"Article validation result: {is_valid} (found {len(keyword_matches)} keywords)")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating article content: {str(e)}")
            return False
    
    def _categorize_article(self, content: str) -> str:
        """Categorize article based on content with improved accuracy."""
        try:
            content = content.lower()
            
            # Count keyword matches for each category
            matches = {
                category: sum(1 for keyword in info["keywords"] if keyword in content)
                for category, info in self.category_mapping.items()
            }
            
            # Get category with most matches
            best_category = max(matches.items(), key=lambda x: x[1])
            
            # Only return category if it has sufficient matches
            if best_category[1] >= 2:
                return best_category[0]
                
            # Default to general if no strong category match
            return "Economy"
            
        except Exception as e:
            logger.error(f"Error categorizing article: {str(e)}")
            return "Economy"  # Default to Economy on error
    
    def _cache_articles(self, articles: List[Dict[str, Any]], cache_file: Path):
        """Cache articles to file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(articles, f, indent=2)
            logger.info(f"Cached {len(articles)} articles to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching articles: {str(e)}")
    
    def get_cached_articles(self, category: str, days: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached articles if available."""
        cache_file = self.cache_dir / f"{category}_{days}d.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
        return None
    
    def prepare_article_for_rag(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare an article for RAG system."""
        try:
            # Split content into chunks (sentences or paragraphs)
            content_chunks = self._split_content(article['content'])
            
            # Create RAG document
            rag_doc = {
                'title': article['title'],
                'source': article['source'],
                'url': article['url'],
                'published_at': article['published_at'],
                'category': article['category'],
                'chunks': content_chunks,
                'metadata': {
                    'source': article['source'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'category': article['category']
                }
            }
            
            # Save to RAG documents directory
            self._save_rag_document(rag_doc)
            
            return rag_doc
        except Exception as e:
            logger.error(f"Error preparing article for RAG: {str(e)}")
            return None
    
    def _split_content(self, content: str) -> List[str]:
        """Split content into meaningful chunks."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        for paragraph in paragraphs:
            # If paragraph is too long, split into sentences
            if len(paragraph) > 500:
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                chunks.extend(sentences)
            else:
                chunks.append(paragraph)
        
        return chunks
    
    def _save_rag_document(self, rag_doc: Dict[str, Any]):
        """Save RAG document to file."""
        try:
            # Create category directory
            category_dir = self.rag_documents_dir / rag_doc['category']
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename from title and timestamp
            safe_title = "".join(c if c.isalnum() else "_" for c in rag_doc['title'])
            timestamp = rag_doc['published_at'].replace(':', '-').replace('+', '-')
            filename = f"{safe_title}_{timestamp}.json"
            
            # Save document
            with open(category_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(rag_doc, f, indent=2)
            
            logger.info(f"Saved RAG document: {filename}")
        except Exception as e:
            logger.error(f"Error saving RAG document: {str(e)}")

async def main():
    try:
        # Initialize handler
        handler = NewsAPIHandler()
        
        # Fetch science articles
        articles = await handler.fetch_articles(
            category="science",
            days=1,
            max_articles=5
        )
        
        # Print results
        for article in articles:
            logger.info(f"\nTitle: {article['title']}")
            logger.info(f"Source: {article['source']}")
            logger.info(f"Category: {article['category']}")
            logger.info(f"Content snippet: {article['content'][:200]}...")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 