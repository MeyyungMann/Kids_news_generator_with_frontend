import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote_plus

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
        Search for articles using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of article dictionaries
        """
        try:
            logger.info(f"Searching for: {query}")
            
            # Construct search URL
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))
            
            # Make request
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find all result elements - try multiple possible class names
            result_elements = soup.find_all(['div', 'article'], class_=['result', 'result__body', 'article'])
            
            for result in result_elements:
                try:
                    # Extract title and link - try multiple possible class names
                    title_elem = result.find(['a', 'h2'], class_=['result__a', 'result__title', 'article__title'])
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href')
                    
                    # Extract snippet - try multiple possible class names
                    snippet_elem = result.find(['a', 'div'], class_=['result__snippet', 'result__description', 'article__description'])
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Extract source - try multiple possible class names
                    source_elem = result.find(['a', 'span'], class_=['result__url', 'result__domain', 'article__source'])
                    source = source_elem.get_text(strip=True) if source_elem else "Unknown Source"
                    
                    # Get full article content
                    content = self._extract_article_content(url) if url else ""
                    
                    # Categorize article
                    category = self._categorize_article(title + " " + snippet)
                    
                    results.append({
                        'title': title,
                        'source': source,
                        'url': url,
                        'content': content,
                        'category': category,
                        'published_at': None  # DuckDuckGo doesn't provide this
                    })
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing result: {str(e)}")
                    continue
            
            logger.info(f"Found {len(results)} articles")
            return results
            
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            return []
    
    def _extract_article_content(self, url: str) -> str:
        """Extract content from article URL."""
        try:
            # Add random delay
            time.sleep(random.uniform(1, 2))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs)
            
            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "..."
            
            return content
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {str(e)}")
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
                'weight': 0.8  # Lower weight for common tech terms
            },
            'health': {
                'keywords': ['health', 'medical', 'medicine', 'disease', 'treatment', 'doctor', 'patient', 'mental health', 'therapy', 'psychologist', 'hospital', 'clinic', 'vaccine', 
    'symptoms', 'diagnosis', 'public health', 'nutrition', 'exercise', 'wellness', 
    'nurse', 'epidemic', 'pandemic', 'infection', 'surgery'],
                'weight': 1.2  # Higher weight for health terms
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
        
        # First pass: count keyword matches
        for category, info in categories.items():
            matches = sum(1 for keyword in info['keywords'] if keyword in text)
            score = matches * info['weight']
            
            # Additional context checks
            if category == 'health' and any(term in text for term in ['mental', 'psychology', 'therapy', 'counseling']):
                score *= 1.5  # Boost health score for mental health content
                
            if category == 'technology' and any(term in text for term in ['parenting', 'family', 'children', 'education']):
                score *= 0.5  # Reduce tech score for family/education content
                
            if score > max_score:
                max_score = score
                best_category = category
        
        # Only return category if it has sufficient matches
        if max_score >= 2:
            return best_category
        
        return 'general' 