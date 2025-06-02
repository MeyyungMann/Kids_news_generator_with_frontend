import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote_plus
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('web_search.log')
    ]
)
logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self):
        """Initialize the web searcher."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search_articles(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles using DuckDuckGo or process direct URLs.
        
        Args:
            query: Search query or direct URL
            max_results: Maximum number of results to return
            
        Returns:
            List of article dictionaries
        """
        try:
            logger.info(f"Processing query/URL: {query}")
            
            # Check if the input is a URL
            if query.startswith(('http://', 'https://')):
                try:
                    # Direct URL processing
                    content = self._extract_article_content(query)
                    if content:
                        # Extract title from URL or content
                        title = query.split('/')[-1].replace('-', ' ').title()
                        category = self._categorize_article(title + " " + content)
                        
                        return [{
                            'title': title,
                            'source': query.split('/')[2],
                            'url': query,
                            'content': content,
                            'category': category,
                            'published_at': None
                        }]
                    else:
                        logger.error(f"Failed to extract content from URL: {query}")
                        return []
                except Exception as e:
                    logger.error(f"Error processing URL {query}: {str(e)}")
                    return []
            
            # If not a URL, proceed with DuckDuckGo search
            try:
                results = self._search_duckduckgo(query, max_results)
                logger.info(f"Found {len(results)} articles")
                return results
            except Exception as e:
                logger.error(f"Error in DuckDuckGo search: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's HTML search."""
        try:
            # Use DuckDuckGo's HTML search instead of API
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            logger.info(f"Making request to DuckDuckGo HTML search: {search_url}")
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find all result elements
            result_elements = soup.find_all('div', class_='result')
            logger.info(f"Number of results found: {len(result_elements)}")
            
            for result in result_elements:
                try:
                    # Extract title and URL
                    title_elem = result.find('a', class_='result__a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    url = title_elem['href']
                    
                    # Get snippet
                    snippet_elem = result.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    logger.info(f"Found article: {title} - {url}")
                    
                    # Get full article content
                    content = self._extract_article_content(url)
                    
                    # Categorize article
                    category = self._categorize_article(title + " " + content)
                    
                    results.append({
                        'title': title,
                        'source': url.split('/')[2],
                        'url': url,
                        'content': content,
                        'category': category,
                        'published_at': None
                    })
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []
    
    def _extract_article_content(self, url: str) -> str:
        """Extract content from article URL."""
        try:
            # Add random delay
            time.sleep(random.uniform(1, 2))
            
            logger.info(f"Fetching content from URL: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Try to find the main article content
            article = soup.find('article') or soup.find('main') or soup.find('div', class_=['article', 'content', 'post'])
            
            if article:
                # Extract text from paragraphs within the article
                paragraphs = article.find_all('p')
                content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                logger.info(f"Found article content with {len(paragraphs)} paragraphs")
            else:
                # Fallback to all paragraphs if no article container found
                paragraphs = soup.find_all('p')
                content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                logger.info(f"Using fallback content with {len(paragraphs)} paragraphs")
            
            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "..."
            
            if not content:
                logger.warning(f"No content found for URL: {url}")
                return ""
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""
    
    def _categorize_article(self, text: str) -> str:
        """Categorize article based on content with improved accuracy."""
        categories = {
            'science': {
                'keywords': ['science', 'research', 'study', 'discovery', 'scientist', 'experiment', 'laboratory', 'physics', 'chemistry', 'biology', 'astronomy', 'genetics', 'microscope', 'data', 'analysis',
    'theory', 'hypothesis', 'observation', 'peer-reviewed'],
                'weight': 1.0
            },
            'technology': {
                'keywords': ['tech', 'technology', 'digital', 'computer', 'software', 'hardware', 'app', 'platform',
    'AI', 'artificial intelligence', 'machine learning', 'robotics', 'smartphone', 'gadget',
    'internet', 'cybersecurity', 'cloud', 'data science', 'programming', 'code', 'startup', 'innovation'
],
                'weight': 0.8
            },
            'health': {
                'keywords': ['health', 'medical', 'medicine', 'disease', 'treatment', 'doctor', 'patient', 'mental health', 'therapy', 'psychologist', 'hospital', 'clinic', 'vaccine', 
    'symptoms', 'diagnosis', 'public health', 'nutrition', 'exercise', 'wellness', 
    'nurse', 'epidemic', 'pandemic', 'infection', 'surgery'],
                'weight': 1.2
            },
            'environment': {
                'keywords': ['environment', 'climate', 'nature', 'earth', 'pollution', 'sustainability', 'green',
    'global warming', 'carbon', 'recycle', 'biodiversity', 'conservation', 'wildlife',
    'eco-friendly', 'deforestation', 'renewable', 'fossil fuels', 'solar', 'wind', 
    'natural disaster', 'climate change', 'greenhouse gas'],
                'weight': 1.0
            },
            'economy': {
                'keywords': ['economy', 'business', 'market', 'finance', 'money', 'investment', 'stock',
    'trade', 'inflation', 'recession', 'budget', 'tax', 'bank', 'interest rate',
    'cryptocurrency', 'bitcoin', 'employment', 'job market', 'GDP', 'economic growth'
],
                'weight': 1.0
            }
        }
        
        text = text.lower()
        max_score = 0
        best_category = 'general'
        
        for category, info in categories.items():
            matches = sum(1 for keyword in info['keywords'] if keyword in text)
            score = matches * info['weight']
            
            if category == 'health' and any(term in text for term in ['mental', 'psychology', 'therapy', 'counseling']):
                score *= 1.5
                
            if category == 'technology' and any(term in text for term in ['parenting', 'family', 'children', 'education']):
                score *= 0.5
                
            if score > max_score:
                max_score = score
                best_category = category
        
        if max_score >= 2:
            return best_category
        
        return 'general' 