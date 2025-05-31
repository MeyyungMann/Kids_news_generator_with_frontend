import logging
from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Example:
    topic: str
    age_group: int
    context: Dict[str, Any]
    original_article: str
    kid_friendly_text: str
    age_appropriateness: float
    engagement: float

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create specific log files
REINFORCEMENT_LOG_FILE = LOGS_DIR / "reinforcement_learning.log"
METRICS_LOG_FILE = LOGS_DIR / "metrics_history.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(REINFORCEMENT_LOG_FILE))
    ]
)

# Create logger
reinforcement_logger = logging.getLogger('reinforcement_learning')

class ReinforcementLogger:
    """Logger for reinforcement learning metrics and events."""
    
    def __init__(self):
        """Initialize the reinforcement learning logger."""
        self.metrics_history = []
        self._load_metrics_history()
    
    def _load_metrics_history(self):
        """Load metrics history from file."""
        try:
            if METRICS_LOG_FILE.exists():
                with open(METRICS_LOG_FILE, 'r') as f:
                    self.metrics_history = json.load(f)
                reinforcement_logger.info(f"Loaded {len(self.metrics_history)} metrics entries")
        except Exception as e:
            reinforcement_logger.error(f"Error loading metrics history: {str(e)}")
            self.metrics_history = []
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        try:
            with open(METRICS_LOG_FILE, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            reinforcement_logger.info("Metrics history saved successfully")
        except Exception as e:
            reinforcement_logger.error(f"Error saving metrics history: {str(e)}")
    
    def log_reward(self, reward: float, iteration: int, context: Dict[str, Any]):
        """Log a reward value."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": "reward",
                "reward": reward,
                "iteration": iteration,
                "context": context
            }
            
            self.metrics_history.append(log_entry)
            self._save_metrics_history()
            
            reinforcement_logger.info(f"Reward: {reward:.4f} (Iteration {iteration})")
            reinforcement_logger.debug(f"Context: {json.dumps(context, indent=2)}")
            
        except Exception as e:
            reinforcement_logger.error(f"Error logging reward: {str(e)}")
    
    def log_model_update(self, loss: float, reward: float, parameters: Dict[str, Any]):
        """Log model update information."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": "model_update",
                "loss": loss,
                "reward": reward,
                "parameters": parameters
            }
            
            self.metrics_history.append(log_entry)
            self._save_metrics_history()
            
            reinforcement_logger.info(f"Model Update - Loss: {loss:.4f}, Reward: {reward:.4f}")
            reinforcement_logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")
            
        except Exception as e:
            reinforcement_logger.error(f"Error logging model update: {str(e)}")
    
    def log_improvement(self, 
                       current_reward: float, 
                       previous_reward: float, 
                       improvement: float,
                       iteration: int):
        """Log improvement metrics."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": "improvement",
                "current_reward": current_reward,
                "previous_reward": previous_reward,
                "improvement": improvement,
                "iteration": iteration
            }
            
            self.metrics_history.append(log_entry)
            self._save_metrics_history()
            
            reinforcement_logger.info(
                f"Improvement - Current: {current_reward:.4f}, "
                f"Previous: {previous_reward:.4f}, "
                f"Delta: {improvement:.4f} "
                f"(Iteration {iteration})"
            )
            
        except Exception as e:
            reinforcement_logger.error(f"Error logging improvement: {str(e)}")
    
    def log_dspy_example(self, example: Example, impact: Dict[str, float]):
        """Log a new DSPy example and its impact."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": "dspy_example",
                "example": {
                    "topic": example.topic,
                    "age_group": example.age_group,
                    "context": example.context,
                    "original_article": example.original_article,
                    "kid_friendly_text": example.kid_friendly_text,
                    "age_appropriateness": example.age_appropriateness,
                    "engagement": example.engagement
                },
                "impact": impact
            }
            
            self.metrics_history.append(log_entry)
            self._save_metrics_history()
            
            reinforcement_logger.info(f"New DSPy Example Added - Topic: {example.topic}")
            reinforcement_logger.debug(f"Impact: {json.dumps(impact, indent=2)}")
            
        except Exception as e:
            reinforcement_logger.error(f"Error logging DSPy example: {str(e)}")

# Create singleton instance
reinforcement_logger_instance = ReinforcementLogger() 