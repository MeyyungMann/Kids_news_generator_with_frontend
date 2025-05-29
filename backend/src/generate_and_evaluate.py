import logging
import asyncio
from typing import List, Dict, Any
from news_api_handler import NewsAPIHandler
from mixtral_basic import MixtralHandler
from evaluator import NewsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def generate_kid_friendly_news(
    news_handler: NewsAPIHandler,
    mixtral_handler: MixtralHandler,
    category: str = "science",
    age_group: int = 8,
    max_articles: int = 5
) -> List[Dict[str, Any]]:
    """Generate kid-friendly versions of news articles."""
    try:
        # Fetch articles
        logger.info(f"Fetching {category} articles...")
        articles = await news_handler.fetch_articles(
            category=category,
            days=1,
            max_articles=max_articles
        )
        
        # Generate kid-friendly versions
        kid_friendly_articles = []
        for article in articles:
            # Create prompt for kid-friendly version
            prompt = f"""Create a kid-friendly version of this news article for children aged {age_group}:
            
            Original Article:
            {article['content']}
            
            Make it:
            1. Easy to understand
            2. Educational
            3. Engaging
            4. Age-appropriate
            
            Kid-friendly version:"""
            
            # Generate kid-friendly content
            result = mixtral_handler.generate_text(prompt)
            
            kid_friendly_articles.append({
                'original_content': article['content'],
                'kid_friendly_content': result['text'],
                'title': article['title'],
                'source': article['source'],
                'category': article['category']
            })
        
        return kid_friendly_articles
        
    except Exception as e:
        logger.error(f"Error generating kid-friendly news: {str(e)}")
        raise

async def main():
    try:
        # Initialize components
        news_handler = NewsAPIHandler()
        mixtral_handler = MixtralHandler()
        evaluator = NewsEvaluator()
        
        # Generate kid-friendly articles
        articles = await generate_kid_friendly_news(
            news_handler=news_handler,
            mixtral_handler=mixtral_handler,
            category="science",
            age_group=8,
            max_articles=3
        )
        
        # Evaluate articles
        results = evaluator.evaluate_articles(articles, age_group=8)
        
        # Print results
        logger.info("\nEvaluation Results:")
        for metric, score in results.items():
            logger.info(f"{metric}: {score:.2f}")
        
        # Plot and save results
        evaluator.plot_results(results)
        evaluator.save_results(results)
        
        # Print sample articles
        logger.info("\nSample Kid-Friendly Articles:")
        for article in articles:
            logger.info("\n" + "="*80)
            logger.info(f"Title: {article['title']}")
            logger.info(f"Source: {article['source']}")
            logger.info(f"Category: {article['category']}")
            logger.info("\nOriginal Content:")
            logger.info(article['original_content'][:200] + "...")
            logger.info("\nKid-Friendly Version:")
            logger.info(article['kid_friendly_content'][:200] + "...")
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 