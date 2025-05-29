import torch
from ml_pipeline import KidsNewsGenerator, RAGConfig, ClassifierConfig
import logging
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineEvaluator:
    def __init__(
        self,
        generator: KidsNewsGenerator,
        test_data: List[Dict[str, Any]]
    ):
        """
        Initialize pipeline evaluator.
        
        Args:
            generator: KidsNewsGenerator instance
            test_data: List of test cases
        """
        self.generator = generator
        self.test_data = test_data
        self.results = []
    
    def evaluate_rag(self) -> Dict[str, float]:
        """Evaluate RAG system performance."""
        try:
            relevance_scores = []
            
            for test_case in tqdm(self.test_data, desc="Evaluating RAG"):
                # Get retrieved context
                context = self.generator.rag.retrieve(test_case["query"])
                
                # Calculate relevance score (simple overlap)
                relevance = self._calculate_relevance(
                    context,
                    test_case["expected_context"]
                )
                relevance_scores.append(relevance)
            
            return {
                "mean_relevance": np.mean(relevance_scores),
                "std_relevance": np.std(relevance_scores)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RAG: {str(e)}")
            raise
    
    def evaluate_classifier(self) -> Dict[str, float]:
        """Evaluate classifier performance."""
        try:
            predictions = []
            true_labels = []
            
            for test_case in tqdm(self.test_data, desc="Evaluating Classifier"):
                # Get prediction
                score = self.generator.classifier.transform([test_case["text"]])[0]
                prediction = 1 if score > self.generator.classifier_config.threshold else 0
                
                predictions.append(prediction)
                true_labels.append(test_case["is_safe"])
            
            return {
                "accuracy": accuracy_score(true_labels, predictions),
                "precision": precision_score(true_labels, predictions),
                "recall": recall_score(true_labels, predictions),
                "f1": f1_score(true_labels, predictions)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classifier: {str(e)}")
            raise
    
    def evaluate_generation(self) -> Dict[str, Any]:
        """Evaluate text generation performance."""
        try:
            generation_times = []
            safety_scores = []
            
            for test_case in tqdm(self.test_data, desc="Evaluating Generation"):
                start_time = time.time()
                
                result = self.generator.generate_news(
                    topic=test_case["query"],
                    age_group=test_case["age_group"]
                )
                
                end_time = time.time()
                generation_times.append(end_time - start_time)
                safety_scores.append(result["safety_score"])
            
            return {
                "mean_generation_time": np.mean(generation_times),
                "std_generation_time": np.std(generation_times),
                "mean_safety_score": np.mean(safety_scores),
                "std_safety_score": np.std(safety_scores)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating generation: {str(e)}")
            raise
    
    def _calculate_relevance(
        self,
        retrieved_context: List[str],
        expected_context: List[str]
    ) -> float:
        """Calculate relevance score between retrieved and expected context."""
        # Simple word overlap score
        retrieved_words = set(' '.join(retrieved_context).lower().split())
        expected_words = set(' '.join(expected_context).lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(retrieved_words.intersection(expected_words))
        return overlap / len(expected_words)
    
    def plot_results(self, save_path: str = "evaluation_results.png"):
        """Plot evaluation results."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot RAG relevance distribution
            sns.histplot(
                data=pd.DataFrame(self.results),
                x="rag_relevance",
                ax=axes[0, 0]
            )
            axes[0, 0].set_title("RAG Relevance Distribution")
            
            # Plot classifier metrics
            classifier_metrics = pd.DataFrame([
                {
                    "metric": k,
                    "score": v
                }
                for k, v in self.evaluate_classifier().items()
            ])
            sns.barplot(
                data=classifier_metrics,
                x="metric",
                y="score",
                ax=axes[0, 1]
            )
            axes[0, 1].set_title("Classifier Performance")
            
            # Plot generation times
            sns.histplot(
                data=pd.DataFrame(self.results),
                x="generation_time",
                ax=axes[1, 0]
            )
            axes[1, 0].set_title("Generation Time Distribution")
            
            # Plot safety scores
            sns.histplot(
                data=pd.DataFrame(self.results),
                x="safety_score",
                ax=axes[1, 1]
            )
            axes[1, 1].set_title("Safety Score Distribution")
            
            plt.tight_layout()
            plt.savefig(save_path)
            logger.info(f"Results plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to a JSON file."""
        try:
            results = {
                "rag_evaluation": self.evaluate_rag(),
                "classifier_evaluation": self.evaluate_classifier(),
                "generation_evaluation": self.evaluate_generation()
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    # Example test data
    test_data = [
        {
            "query": "why is the sky blue",
            "expected_context": [
                "Rayleigh scattering",
                "sunlight",
                "atmosphere",
                "blue light"
            ],
            "text": "The sky is blue because of how sunlight interacts with our atmosphere.",
            "is_safe": 1,
            "age_group": 8
        },
        # Add more test cases here
    ]
    
    try:
        # Initialize generator
        generator = KidsNewsGenerator()
        
        # Initialize evaluator
        evaluator = PipelineEvaluator(generator, test_data)
        
        # Run evaluations
        logger.info("Evaluating RAG system...")
        rag_results = evaluator.evaluate_rag()
        logger.info(f"RAG Results: {rag_results}")
        
        logger.info("\nEvaluating classifier...")
        classifier_results = evaluator.evaluate_classifier()
        logger.info(f"Classifier Results: {classifier_results}")
        
        logger.info("\nEvaluating generation...")
        generation_results = evaluator.evaluate_generation()
        logger.info(f"Generation Results: {generation_results}")
        
        # Save and plot results
        evaluator.save_results()
        evaluator.plot_results()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 