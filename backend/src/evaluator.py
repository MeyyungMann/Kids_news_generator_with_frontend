import logging
from typing import List, Dict, Any
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from news_api_handler import NewsAPIHandler
import asyncio
from datetime import datetime
from mixtral_basic import GPUCompatibilityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants for folder structure
RESULTS_BASE_DIR = Path("results")
EVALUATION_DIR = RESULTS_BASE_DIR / "evaluations"
EXPORTS_DIR = RESULTS_BASE_DIR / "exports"
DASHBOARD_DIR = RESULTS_BASE_DIR / "dashboard"

class NewsEvaluator:
    def __init__(self):
        """Initialize the news evaluator."""
        # Check device compatibility
        self.device, self.diagnostics = GPUCompatibilityChecker.get_optimal_device()
        
        # Log system diagnostics
        logger.info("System Diagnostics:")
        logger.info(f"OS: {self.diagnostics['system_info']['os']} {self.diagnostics['system_info']['os_version']}")
        logger.info(f"Python: {self.diagnostics['system_info']['python_version']}")
        logger.info(f"CPU Cores: {self.diagnostics['system_info']['cpu_count']}")
        logger.info(f"Available Memory: {self.diagnostics['system_info']['available_memory'] / (1024**3):.1f} GB")
        
        if self.diagnostics['gpu_info']['available']:
            for device in self.diagnostics['gpu_info']['devices']:
                logger.info(f"GPU {device['id']}: {device['name']}")
                logger.info(f"  Memory: {device['memory_free']/1024:.1f} GB free of {device['memory_total']/1024:.1f} GB")
                logger.info(f"  Temperature: {device['temperature']}°C")
                logger.info(f"  Load: {device['load']:.1f}%")
        
        # Load models with appropriate device settings
        logger.info("Loading evaluation models...")
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load readability analyzer
            self.readability_analyzer = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                device=0 if self.device == "cuda" else -1
            )
            
            # Create results directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir = EVALUATION_DIR / timestamp
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created results directory: {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def evaluate_articles(
        self,
        articles: List[Dict[str, Any]],
        age_group: int
    ) -> Dict[str, Any]:
        """
        Evaluate generated kid-friendly articles.
        
        Args:
            articles: List of generated articles
            age_group: Target age group
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Starting evaluation of {len(articles)} articles for age group {age_group}")
            
            results = {
                'readability': [],
                'safety': [],
                'relevance': [],
                'engagement': [],
                'educational_value': []
            }
            
            # Evaluate each article
            for i, article in enumerate(articles, 1):
                logger.info(f"\nEvaluating article {i}/{len(articles)}")
                logger.info(f"Title: {article.get('title', 'Untitled')}")
                
                # Evaluate readability
                readability_score = self._evaluate_readability(
                    article['kid_friendly_content'],
                    age_group
                )
                results['readability'].append(readability_score)
                logger.info(f"Readability score: {readability_score:.2f}")
                
                # Evaluate safety
                safety_score = self._evaluate_safety(
                    article['kid_friendly_content']
                )
                results['safety'].append(safety_score)
                logger.info(f"Safety score: {safety_score:.2f}")
                
                # Evaluate relevance
                relevance_score = self._evaluate_relevance(
                    article['original_content'],
                    article['kid_friendly_content']
                )
                results['relevance'].append(relevance_score)
                logger.info(f"Relevance score: {relevance_score:.2f}")
                
                # Evaluate engagement
                engagement_score = self._evaluate_engagement(
                    article['kid_friendly_content']
                )
                results['engagement'].append(engagement_score)
                logger.info(f"Engagement score: {engagement_score:.2f}")
                
                # Evaluate educational value
                edu_score = self._evaluate_educational_value(
                    article['kid_friendly_content']
                )
                results['educational_value'].append(edu_score)
                logger.info(f"Educational value score: {edu_score:.2f}")
            
            # Calculate average scores
            avg_results = {
                metric: np.mean(scores)
                for metric, scores in results.items()
            }
            
            # Calculate standard deviations
            std_results = {
                metric: np.std(scores)
                for metric, scores in results.items()
            }
            
            # Combine results
            final_results = {
                'averages': avg_results,
                'standard_deviations': std_results,
                'individual_scores': results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_articles': len(articles),
                    'age_group': age_group,
                    'device': self.device,
                    'system_info': self.diagnostics['system_info'],
                    'gpu_info': self.diagnostics['gpu_info'],
                    'cuda_info': self.diagnostics['cuda_info']
                }
            }
            
            logger.info("\nEvaluation Summary:")
            for metric, score in avg_results.items():
                logger.info(f"{metric}: {score:.2f} (±{std_results[metric]:.2f})")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error evaluating articles: {str(e)}")
            raise
    
    def _evaluate_readability(self, text: str, age_group: int) -> float:
        """Evaluate text readability for target age group."""
        try:
            # Simple readability metrics
            words = text.split()
            sentences = text.split('.')
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words])
            
            # Average sentence length
            avg_sentence_length = np.mean([len(sent.split()) for sent in sentences if sent.strip()])
            
            # Calculate readability score (0-1)
            score = 1.0
            if age_group <= 8:
                # For younger children, prefer shorter words and sentences
                score *= (1 - (avg_word_length - 4) / 10)  # Penalize words longer than 4 letters
                score *= (1 - (avg_sentence_length - 5) / 15)  # Penalize sentences longer than 5 words
            else:
                # For older children, allow more complexity
                score *= (1 - (avg_word_length - 6) / 15)
                score *= (1 - (avg_sentence_length - 8) / 20)
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error evaluating readability: {str(e)}")
            return 0.0
    
    def _evaluate_safety(self, text: str) -> float:
        """Evaluate content safety."""
        try:
            result = self.readability_analyzer(text)[0]
            return 1 - result['score'] if result['label'] == 'hate' else result['score']
        except Exception as e:
            logger.error(f"Error evaluating safety: {str(e)}")
            return 0.0
    
    def _evaluate_relevance(
        self,
        original: str,
        kid_friendly: str
    ) -> float:
        """Evaluate content relevance to original article."""
        try:
            # Create embeddings
            original_embedding = self.embedding_model.encode([original])[0]
            kid_friendly_embedding = self.embedding_model.encode([kid_friendly])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [original_embedding],
                [kid_friendly_embedding]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error evaluating relevance: {str(e)}")
            return 0.0
    
    def _evaluate_engagement(self, text: str) -> float:
        """Evaluate content engagement."""
        try:
            # Simple engagement metrics
            words = text.split()
            
            # Check for engaging elements
            has_questions = '?' in text
            has_exclamations = '!' in text
            has_numbers = any(word.isdigit() for word in words)
            has_capitalized = any(word.isupper() and len(word) > 1 for word in words)
            
            # Calculate engagement score
            score = 0.0
            if has_questions: score += 0.25
            if has_exclamations: score += 0.25
            if has_numbers: score += 0.25
            if has_capitalized: score += 0.25
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating engagement: {str(e)}")
            return 0.0
    
    def _evaluate_educational_value(self, text: str) -> float:
        """Evaluate educational value."""
        try:
            # Simple educational value metrics
            words = text.split()
            
            # Check for educational elements
            has_definitions = 'means' in text.lower() or 'is a' in text.lower()
            has_examples = 'for example' in text.lower() or 'such as' in text.lower()
            has_explanations = 'because' in text.lower() or 'why' in text.lower()
            has_facts = any(word.isdigit() for word in words)
            
            # Calculate educational value score
            score = 0.0
            if has_definitions: score += 0.25
            if has_examples: score += 0.25
            if has_explanations: score += 0.25
            if has_facts: score += 0.25
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating educational value: {str(e)}")
            return 0.0
    
    def plot_results(
        self,
        results: Dict[str, Any],
        save_path: str = None
    ):
        """Plot evaluation results."""
        try:
            if save_path is None:
                save_path = self.results_dir / "metrics.png"
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot average scores
            metrics = list(results['averages'].keys())
            scores = list(results['averages'].values())
            stds = list(results['standard_deviations'].values())
            
            # Create bar plot with error bars
            ax1.bar(
                metrics,
                scores,
                yerr=stds,
                capsize=5,
                alpha=0.7
            )
            ax1.set_title('Average Evaluation Scores')
            ax1.set_ylim(0, 1)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot individual scores
            individual_data = pd.DataFrame(results['individual_scores'])
            sns.boxplot(data=individual_data, ax=ax2)
            ax2.set_title('Score Distribution')
            ax2.set_ylim(0, 1)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path)
            logger.info(f"Results plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise
    
    def save_results(
        self,
        results: Dict[str, Any],
        articles: List[Dict[str, Any]],
        filename: str = None
    ):
        """Save evaluation results to a JSON file."""
        try:
            if filename is None:
                filename = self.results_dir / "metrics.json"
            
            # Add article-specific analysis
            article_analysis = []
            for i, article in enumerate(articles):
                analysis = {
                    'title': article.get('title', 'Untitled'),
                    'source': article.get('source', 'Unknown'),
                    'category': article.get('category', 'general'),
                    'scores': {
                        metric: scores[i]
                        for metric, scores in results['individual_scores'].items()
                    },
                    'recommendations': self._generate_recommendations(
                        {metric: scores[i] for metric, scores in results['individual_scores'].items()}
                    )
                }
                article_analysis.append(analysis)
            
            # Add article analysis to results
            results['article_analysis'] = article_analysis
            
            # Save detailed results
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detailed results saved to {filename}")
            
            # Save summary to a separate file
            summary_file = self.results_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Evaluation Summary\n")
                f.write("=================\n\n")
                f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
                f.write(f"Number of articles: {results['metadata']['num_articles']}\n")
                f.write(f"Target age group: {results['metadata']['age_group']}\n")
                f.write(f"Device: {results['metadata']['device']}\n\n")
                
                f.write("System Information:\n")
                f.write(f"OS: {results['metadata']['system_info']['os']} {results['metadata']['system_info']['os_version']}\n")
                f.write(f"Python: {results['metadata']['system_info']['python_version']}\n")
                f.write(f"CPU Cores: {results['metadata']['system_info']['cpu_count']}\n")
                f.write(f"Available Memory: {results['metadata']['system_info']['available_memory'] / (1024**3):.1f} GB\n\n")
                
                if results['metadata']['gpu_info']['available']:
                    f.write("GPU Information:\n")
                    for device in results['metadata']['gpu_info']['devices']:
                        f.write(f"GPU {device['id']}: {device['name']}\n")
                        f.write(f"  Memory: {device['memory_free']/1024:.1f} GB free of {device['memory_total']/1024:.1f} GB\n")
                        f.write(f"  Temperature: {device['temperature']}°C\n")
                        f.write(f"  Load: {device['load']:.1f}%\n")
                    f.write("\n")
                
                f.write("Overall Average Scores:\n")
                for metric, score in results['averages'].items():
                    std = results['standard_deviations'][metric]
                    f.write(f"{metric}: {score:.2f} (±{std:.2f})\n")
                
                f.write("\nArticle-Specific Analysis:\n")
                f.write("=======================\n\n")
                for analysis in article_analysis:
                    f.write(f"Title: {analysis['title']}\n")
                    f.write(f"Source: {analysis['source']}\n")
                    f.write(f"Category: {analysis['category']}\n")
                    f.write("\nScores:\n")
                    for metric, score in analysis['scores'].items():
                        f.write(f"- {metric}: {score:.2f}\n")
                    f.write("\nRecommendations:\n")
                    for rec in analysis['recommendations']:
                        f.write(f"- {rec}\n")
                    f.write("\n" + "="*50 + "\n\n")
            
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on scores."""
        recommendations = []
        
        # Readability recommendations
        if scores['readability'] < 0.7:
            recommendations.append("Consider simplifying language and sentence structure")
        elif scores['readability'] < 0.5:
            recommendations.append("Significant simplification needed for target age group")
        
        # Safety recommendations
        if scores['safety'] < 0.8:
            recommendations.append("Review content for age-appropriate language and topics")
        elif scores['safety'] < 0.6:
            recommendations.append("Content may need significant revision for safety")
        
        # Relevance recommendations
        if scores['relevance'] < 0.7:
            recommendations.append("Ensure key information from original article is preserved")
        elif scores['relevance'] < 0.5:
            recommendations.append("Content may have strayed too far from original article")
        
        # Engagement recommendations
        if scores['engagement'] < 0.6:
            recommendations.append("Add more interactive elements (questions, examples)")
        elif scores['engagement'] < 0.4:
            recommendations.append("Content needs more engaging elements for children")
        
        # Educational value recommendations
        if scores['educational_value'] < 0.6:
            recommendations.append("Include more educational elements (definitions, examples)")
        elif scores['educational_value'] < 0.4:
            recommendations.append("Content needs more educational value for learning")
        
        return recommendations

async def main():
    try:
        # Initialize components
        news_handler = NewsAPIHandler()
        evaluator = NewsEvaluator()
        
        # Fetch real articles
        logger.info("Fetching science articles...")
        articles = await news_handler.fetch_articles(
            category="science",
            days=1,
            max_articles=5
        )
        
        # Prepare articles for evaluation
        evaluation_articles = []
        for article in articles:
            evaluation_articles.append({
                'original_content': article['content'],
                'kid_friendly_content': article['content']  # This will be replaced with actual kid-friendly version
            })
        
        # Evaluate articles
        results = evaluator.evaluate_articles(evaluation_articles, age_group=8)
        
        # Print results
        logger.info("\nEvaluation Results:")
        for metric, score in results['averages'].items():
            logger.info(f"{metric}: {score:.2f} (±{results['standard_deviations'][metric]:.2f})")
        
        # Plot and save results
        evaluator.plot_results(results)
        evaluator.save_results(results, articles)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 