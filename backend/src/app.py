# cd backend/src
# python -m uvicorn app:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from news_api_handler import NewsAPIHandler
from ml_pipeline import KidsNewsGenerator  # Import the KidsNewsGenerator
from image_generator import KidFriendlyImageGenerator  # Import the KidFriendlyImageGenerator
from config import Config  # Import the config
from fastapi.staticfiles import StaticFiles
from web_search import WebSearcher
from clip_handler import CLIPHandler
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Kids News Generator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize handlers
news_handler = None
generator = None
image_generator = None
web_searcher = None
clip_handler = None

# Add this function to create necessary directories
def create_directories():
    """Create necessary directories for the application."""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    images_dir = results_dir / "images"
    summaries_dir = results_dir / "summaries"
    
    # Create category directories
    categories = ["Science", "Technology", "Environment", "Health", "Economy"]
    for category in categories:
        (images_dir / category).mkdir(parents=True, exist_ok=True)
        (summaries_dir / category).mkdir(parents=True, exist_ok=True)
    
    return results_dir, images_dir, summaries_dir

# Call this before creating the FastAPI app
results_dir, images_dir, summaries_dir = create_directories()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global news_handler, generator, image_generator, web_searcher, clip_handler
    try:
        # Validate environment variables
        Config.validate()
        
        # Initialize handlers
        news_handler = NewsAPIHandler()
        generator = KidsNewsGenerator(
            offline_mode=Config.OFFLINE_MODE,
            skip_article_loading=Config.SKIP_ARTICLE_LOADING
        )
        clip_handler = CLIPHandler()  # Initialize CLIP handler first
        image_generator = KidFriendlyImageGenerator(clip_handler=clip_handler)  # Pass the CLIP handler
        web_searcher = WebSearcher()
        
        # Mount static files
        results_dir = Path(__file__).parent.parent / "results"
        images_dir = results_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Keep only the valid categories list
CATEGORIES = ["Science", "Technology", "Health", "Environment", "Economy"]

class GenerateRequest(BaseModel):
    topic: str
    age_group: int
    category: str
    include_glossary: bool = True
    generate_image: bool = True
    original_news: Optional[Dict[str, Any]] = None

class WebSearchRequest(BaseModel):
    query: str

# Initialize web searcher
web_searcher = WebSearcher()

@app.post("/api/generate")
async def generate_article(request: GenerateRequest) -> Dict[str, Any]:
    """Generate a kid-friendly article."""
    try:
        logger.info(f"Generating article for topic: {request.topic}, age_group: {request.age_group}")
        
        # Load articles into RAG system before generation
        logger.info("Loading articles into RAG system...")
        generator.rag.load_news_articles()
        
        # First, fetch a real news article
        articles = await news_handler.fetch_articles(
            category=request.category,
            days=1,
            max_articles=1
        )
        
        if not articles:
            logger.warning(f"No articles found for category: {request.category}")
            # Use the topic directly if no articles found
            topic = request.topic
            original_article = {
                "title": request.topic,
                "content": "",
                "source": "Custom Topic",
                "url": "",
                "published_at": datetime.now().isoformat()
            }
        else:
            original_article = articles[0]
            topic = original_article["title"]
        
        logger.info(f"Using topic for generation: {topic}")
        
        # Generate kid-friendly version
        try:
            # Properly await the async generate_news function
            result = await generator.generate_news(
                topic=topic,
                age_group=request.age_group
            )
            logger.info("Successfully generated news content")
        except Exception as e:
            logger.error(f"Error in generate_news: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")
        
        # Generate image if requested
        image_url = None
        clip_similarity_score = None
        if request.generate_image:
            try:
                image_result = image_generator.generate_image(
                    content={
                        "topic": topic,
                        "text": result["text"],
                        "category": request.category,
                        "timestamp": result["timestamp"]
                    },
                    age_group=request.age_group
                )
                image_url = f"/images/{request.category}/{Path(image_result['image_path']).name}"
                clip_similarity_score = image_result['metadata'].get('clip_similarity_score')
            except Exception as e:
                logger.error(f"Error generating image: {str(e)}")
                # Continue without image if generation fails
        
        # Create a summary object
        summary_data = {
            "title": topic,
            "category": request.category,
            "content": result["text"],
            "timestamp": result["timestamp"],
            "age_group": request.age_group,
            "safety_score": result.get("safety_score", 0.0),
            "original_article": original_article
        }
        
        # Save the summary
        try:
            generator.save_summary(result, summary_data)
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            # Continue even if saving fails
        
        # Prepare the response
        response = {
            "title": topic,
            "content": result["text"],
            "category": request.category,
            "reading_level": f"Ages {request.age_group}-{request.age_group + 3}",
            "safety_score": result.get("safety_score", 0.0),
            "image_url": image_url,
            "clip_similarity_score": clip_similarity_score,
            "original_article": original_article
        }
        
        logger.info(f"Successfully generated article with safety score: {response['safety_score']}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating article: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint to verify server is running."""
    return {"message": "Server is running", "categories": CATEGORIES}

@app.get("/api/news/{category}")
async def get_news(category: str) -> Dict[str, Any]:
    """Get real news articles for a specific category."""
    logger.info(f"Received request for category: {category}")
    
    try:
        # Validate category
        if category not in CATEGORIES:
            return {"articles": []}  # Return empty list for invalid category
        
        # Fetch articles using NewsAPIHandler
        articles = await news_handler.fetch_articles(
            category=category,
            days=1,
            max_articles=5
        )
        
        logger.info(f"Fetched {len(articles) if articles else 0} articles")
        
        # Process articles to match frontend expectations
        processed_articles = []
        for article in articles:
            try:
                processed_article = {
                    "id": article.get("id", ""),
                    "title": article.get("title", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                    "content": article.get("content", ""),
                    "category": category,
                    "description": article.get("description", ""),
                    "image_url": article.get("urlToImage", None)
                }
                processed_articles.append(processed_article)
            except Exception as e:
                logger.error(f"Error processing article: {str(e)}")
                continue
        
        return {"articles": processed_articles}
        
    except Exception as e:
        logger.error(f"Error fetching news for category {category}: {str(e)}")
        return {"articles": []}  # Return empty list instead of raising error

@app.get("/api/news/{category}/full/{article_id}")
async def get_full_article(category: str, article_id: str) -> Dict[str, Any]:
    """Get full content of a specific article."""
    try:
        # Validate category
        if category not in CATEGORIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {', '.join(CATEGORIES)}"
            )
        
        # Fetch articles
        articles = await news_handler.fetch_articles(
            category=category,
            days=1,
            max_articles=5
        )
        
        # Find the specific article
        article = next((a for a in articles if a.get("id") == article_id), None)
        
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return {
            "id": article_id,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "content": article.get("content", ""),
            "category": category,
            "description": article.get("description", ""),
            "image_url": article.get("urlToImage", None)
        }
        
    except Exception as e:
        logger.error(f"Error fetching full article {article_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories() -> List[str]:
    """Get list of available categories."""
    return CATEGORIES

@app.get("/api/articles/history")
async def get_article_history(category: str = "all", age_group: str = "all") -> Dict[str, Any]:
    """Get history of generated articles."""
    try:
        logger.info(f"Fetching article history for category: {category}, age_group: {age_group}")
        
        # Get the summaries directory
        summaries_dir = Path(__file__).parent.parent / "results" / "summaries"
        feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
        
        # Initialize list to store articles
        articles = []
        
        # If category is "all", search in all category directories
        if category.lower() == "all":
            categories = ["Science", "Technology", "Environment", "Health", "Economy"]
        else:
            categories = [category]
        
        # Search for summary files in each category directory
        for cat in categories:
            category_dir = summaries_dir / cat
            if not category_dir.exists():
                continue
                
            # Look for summary files
            for summary_file in category_dir.glob("*.json"):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                        
                        # Filter by age group if specified
                        if age_group != "all":
                            article_age_group = summary_data.get('age_group', 7)
                            if str(article_age_group) != age_group.split('-')[0]:
                                continue
                        
                        # Get corresponding image if it exists
                        image_path = None
                        image_dir = Path(__file__).parent.parent / "results" / "images" / cat
                        image_file = image_dir / f"{summary_file.stem}.png"
                        if image_file.exists():
                            image_path = f"/images/{cat}/{image_file.name}"
                        
                        # Get feedback for this article
                        feedback_data = None
                        feedback_category_dir = feedback_dir / cat
                        if feedback_category_dir.exists():
                            # Look for feedback files that match the article ID
                            feedback_files = list(feedback_category_dir.glob(f"{summary_file.stem}*.json"))
                            if feedback_files:
                                # Get the most recent feedback
                                latest_feedback = max(feedback_files, key=lambda x: x.stat().st_mtime)
                                with open(latest_feedback, 'r', encoding='utf-8') as f:
                                    feedback_data = json.load(f)
                        
                        # Create article object
                        article = {
                            "topic": summary_data.get("topic", ""),
                            "text": summary_data.get("text", summary_data.get("prompt", "")),  # Try both text and prompt
                            "timestamp": summary_data.get("timestamp", ""),
                            "image_url": image_path,
                            "age_group": f"{summary_data.get('age_group', 7)}-{summary_data.get('age_group', 7) + 3}",
                            "combined_score": summary_data.get("safety_score", 0.0),
                            "original_article": summary_data.get("original_article", {}),
                            "feedback": feedback_data if feedback_data else None,
                            "is_favorite": summary_data.get("is_favorite", False)  # Add favorite status
                        }
                        articles.append(article)
                except Exception as e:
                    logger.error(f"Error reading summary file {summary_file}: {str(e)}")
                    continue
        
        # Sort articles by timestamp (newest first)
        articles.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        logger.info(f"Found {len(articles)} articles in history")
        return {"articles": articles}
        
    except Exception as e:
        logger.error(f"Error fetching article history: {str(e)}")
        return {"articles": []}

@app.post("/api/update-rag")
async def update_rag_system():
    """Update the RAG system with new articles."""
    try:
        logger.info("Updating RAG system with new articles...")
        for category in CATEGORIES:
            try:
                articles = await news_handler.fetch_articles(
                    category=category,
                    days=1,
                    max_articles=1
                )
                
                if articles:
                    logger.info(f"Fetched article for {category}: {articles[0]['title']}")
                    # Prepare article for RAG
                    rag_doc = news_handler.prepare_article_for_rag(articles[0])
                    if rag_doc:
                        logger.info(f"Prepared article for RAG: {articles[0]['title']}")
                else:
                    logger.warning(f"No articles found for category: {category}")
            except Exception as e:
                logger.error(f"Error fetching article for {category}: {str(e)}")
        
        # Reload articles into RAG system
        generator.rag.load_news_articles()
        
        return {"message": "RAG system updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-articles")
async def search_articles(request: WebSearchRequest) -> Dict[str, Any]:
    """Search for articles using web search."""
    try:
        logger.info(f"Searching for articles with query: {request.query}")
        articles = web_searcher.search_articles(request.query, max_results=5)
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Error searching articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-from-url")
async def generate_from_url(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a kid-friendly article from a URL."""
    try:
        logger.info("Received generate-from-url request")
        logger.info(f"Request data: {request}")
        
        url = request.get("url")
        age_group = request.get("age_group")
        
        if not url:
            logger.error("URL is missing from request")
            raise HTTPException(status_code=400, detail="URL is required")
        if not age_group:
            logger.error("Age group is missing from request")
            raise HTTPException(status_code=400, detail="Age group is required")
        
        logger.info(f"Processing URL: {url} for age group: {age_group}")
        
        # Load articles into RAG system before generation
        logger.info("Loading articles into RAG system...")
        generator.rag.load_news_articles()
        
        # Extract article content
        try:
            content = web_searcher._extract_article_content(url)
            if not content:
                logger.error(f"Could not extract content from URL: {url}")
                raise HTTPException(status_code=400, detail="Could not extract content from URL")
            logger.info(f"Successfully extracted content, length: {len(content)}")
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting content: {str(e)}")
        
        # Create a custom article object
        custom_article = {
            "title": request.get("title", "Custom Article"),
            "content": content,
            "source": request.get("source", "Web Search"),
            "url": url,
            "category": web_searcher._categorize_article(content),
            "published_at": datetime.now().isoformat()
        }
        
        # Generate kid-friendly content
        try:
            # Properly await the async generate_news function
            result = await generator.generate_news(
                topic=custom_article["title"],
                age_group=age_group
            )
            logger.info("Successfully generated kid-friendly content")
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")
        
        # Generate image if requested
        image_url = None
        clip_similarity_score = None
        if request.get("generate_image", False):
            try:
                image_result = image_generator.generate_image(
                    content={
                        "topic": custom_article["title"],
                        "text": result["text"],
                        "category": custom_article["category"],
                        "timestamp": result["timestamp"]
                    },
                    age_group=age_group
                )
                image_url = f"/images/{custom_article['category']}/{Path(image_result['image_path']).name}"
                clip_similarity_score = image_result['metadata'].get('clip_similarity_score')
            except Exception as e:
                logger.error(f"Error generating image: {str(e)}")
                # Continue without image if generation fails
        
        # Save the summary
        try:
            summary_data = {
                "title": custom_article["title"],
                "category": custom_article["category"],
                "content": result["text"],
                "timestamp": result["timestamp"],
                "age_group": age_group,
                "safety_score": result.get("safety_score", 0.0),
                "original_article": custom_article
            }
            generator.save_summary(result, summary_data)
            logger.info("Successfully saved summary")
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            # Continue even if saving fails
        
        # Create response
        response = {
            "title": custom_article["title"],
            "content": result["text"],
            "category": custom_article["category"],
            "reading_level": f"Ages {age_group}-{age_group + 3}",
            "safety_score": result.get("safety_score", 0.0),
            "image_url": image_url,
            "clip_similarity_score": clip_similarity_score,
            "original_article": custom_article
        }
        
        logger.info("Successfully prepared response")
        return response
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in generate-from-url: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate-from-url: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compute-similarity")
async def compute_similarity(request: Dict[str, Any]) -> Dict[str, Any]:
    """Compute similarity between an image and text."""
    try:
        image_path = request.get("image_path")
        text = request.get("text")
        
        if not image_path or not text:
            raise HTTPException(status_code=400, detail="Both image_path and text are required")
        
        similarity = clip_handler.compute_similarity(image_path, text)
        
        return {
            "similarity": similarity,
            "image_path": image_path,
            "text": text
        }
        
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add Feedback model
class FeedbackItem(BaseModel):
    feedback_type: str
    rating: int  # 1-5
    comments: Optional[str] = None

class FeedbackRequest(BaseModel):
    article_id: str
    age_group: int
    category: str
    feedback: List[FeedbackItem]

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    """Submit user feedback for an article."""
    try:
        logger.info(f"Received feedback for article {request.article_id}")
        
        # Create feedback directory if it doesn't exist
        feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directory
        category_dir = feedback_dir / request.category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create feedback file
        feedback_file = category_dir / f"{request.article_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare feedback data
        feedback_data = {
            "article_id": request.article_id,
            "age_group": request.age_group,
            "category": request.category,
            "timestamp": datetime.now().isoformat(),
            "feedback": [
                {
                    "feedback_type": item.feedback_type,
                    "rating": item.rating,
                    "comments": item.comments
                }
                for item in request.feedback
            ]
        }
        
        # Save feedback
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
        
        # Update the generator with feedback
        generator.update_with_feedback(feedback_data)
        
        return {"message": "Feedback submitted successfully", "feedback_id": feedback_file.stem}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/history")
async def get_feedback_history(
    category: Optional[str] = None,
    age_group: Optional[int] = None,
    feedback_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get feedback history with optional filters."""
    try:
        feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
        if not feedback_dir.exists():
            return {"feedback": []}

        all_feedback = []
        
        # Get all category directories
        category_dirs = [feedback_dir / cat for cat in os.listdir(feedback_dir) 
                        if (feedback_dir / cat).is_dir()]
        
        # If category is specified, only look in that category
        if category:
            category_dirs = [feedback_dir / category]
        
        # Process each category directory
        for cat_dir in category_dirs:
            if not cat_dir.exists():
                continue
                
            # Read all feedback files in the category
            for feedback_file in cat_dir.glob("*.json"):
                try:
                    with open(feedback_file, "r") as f:
                        feedback_data = json.load(f)
                        
                        # Apply filters
                        if age_group and feedback_data.get("age_group") != age_group:
                            continue
                        if feedback_type and feedback_data.get("feedback_type") != feedback_type:
                            continue
                            
                        all_feedback.append(feedback_data)
                except Exception as e:
                    logger.error(f"Error reading feedback file {feedback_file}: {str(e)}")
                    continue
        
        # Sort feedback by timestamp (newest first)
        all_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "feedback": all_feedback,
            "total_count": len(all_feedback),
            "categories": [cat.name for cat in category_dirs],
            "age_groups": sorted(list(set(f.get("age_group") for f in all_feedback))),
            "feedback_types": sorted(list(set(f.get("feedback_type") for f in all_feedback)))
        }
        
    except Exception as e:
        logger.error(f"Error retrieving feedback history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/articles/{category}/{filename}")
async def delete_article(category: str, filename: str):
    """Delete an article from the history."""
    try:
        # Construct the path to the article file
        article_path = Path(__file__).parent.parent / "results" / "summaries" / category / filename
        
        # Check if file exists
        if not article_path.exists():
            raise HTTPException(status_code=404, detail="Article not found")
        
        # Delete the article file
        article_path.unlink()
        
        # Also try to delete associated image if it exists
        image_path = Path(__file__).parent.parent / "results" / "images" / category / f"{article_path.stem}.png"
        if image_path.exists():
            image_path.unlink()
        
        logger.info(f"Successfully deleted article: {filename}")
        return {"message": "Article deleted successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/articles/{category}/{filename}/favorite")
async def toggle_favorite(category: str, filename: str) -> Dict[str, Any]:
    """Toggle the favorite status of an article."""
    try:
        # Get the article ID from the filename
        article_id = Path(filename).stem
        
        # Toggle favorite status using the generator
        is_favorite = generator.toggle_favorite(article_id, category)
        
        return {"is_favorite": is_favorite}
        
    except FileNotFoundError as e:
        logger.error(f"Article not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error toggling favorite status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT
    ) 