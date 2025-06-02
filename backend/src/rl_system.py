import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import logging
from datetime import datetime
from collections import deque
import random
from reinforcement_logger import reinforcement_logger_instance, Example
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentState:
    def __init__(self, age_group: int, category: str, topic: str, content_features: Dict[str, float]):
        self.age_group = age_group
        self.category = category
        self.topic = topic
        self.features = content_features
        self.timestamp = datetime.now()

    def to_tensor(self) -> torch.Tensor:
        # Convert state to tensor representation
        feature_vector = [
            self.age_group / 18.0,  # Normalize age group
            *[self.features.get(f, 0.0) for f in [
                'vocabulary_complexity',
                'sentence_length',
                'interactive_elements',
                'example_count',
                'question_count'
            ]]
        ]
        return torch.FloatTensor(feature_vector)

class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: ContentState, action: int, reward: float, next_state: ContentState, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

class RLSystem:
    def __init__(self, state_size: int = 6, action_size: int = 10, hidden_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Initialize Q-Networks
        self.q_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize replay buffer with smaller capacity
        self.replay_buffer = ReplayBuffer(capacity=1000)  # Reduced from 10000
        
        # RL parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32  # Reduced from 64
        self.update_frequency = 20  # Increased from 10
        self.steps = 0
        
        # Load existing feedback data
        self.feedback_dir = Path(__file__).parent.parent / "data" / "feedback"
        self.feedback_data = self._load_feedback_data()
        
        # Initialize metrics history
        self.metrics_history = []
        
        # Initialize feature cache
        self._feature_cache = {}
        self._feedback_cache = {}
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)

    def _evaluate_content(self, content: str, age_group: int) -> float:
        """Evaluate content using feedback-based metrics."""
        try:
            # Extract features from content
            features = self.get_content_features(content)
            
            # Get feedback insights for this age group
            feedback_insights = self._get_feedback_insights(age_group)
            
            # Calculate score based on features and feedback
            score = 0.0
            weights = {
                'vocabulary_complexity': 0.3,
                'sentence_length': 0.2,
                'interactive_elements': 0.2,
                'example_count': 0.15,
                'question_count': 0.15
            }
            
            for feature, weight in weights.items():
                if feature in features:
                    score += features[feature] * weight
            
            # Adjust score based on feedback insights
            if feedback_insights['age_appropriate'] > 0:
                score *= (1 + feedback_insights['age_appropriate'] * 0.1)
            if feedback_insights['engagement'] > 0:
                score *= (1 + feedback_insights['engagement'] * 0.1)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error evaluating content: {str(e)}")
            return 0.0

    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load and process feedback data for RL training."""
        feedback_data = {
            "age_appropriate": {},
            "engagement": {},
            "clarity": {}
        }
        
        try:
            for category_dir in self.feedback_dir.glob("*"):
                if not category_dir.is_dir():
                    continue
                    
                for feedback_file in category_dir.glob("*.json"):
                    try:
                        with open(feedback_file, "r") as f:
                            feedback = json.load(f)
                            
                            # Process feedback for RL
                            age_group = str(feedback["age_group"])
                            
                            # Handle nested feedback array
                            for feedback_item in feedback.get("feedback", []):
                                feedback_type = feedback_item["feedback_type"]
                                
                                if age_group not in feedback_data[feedback_type]:
                                    feedback_data[feedback_type][age_group] = {
                                        "ratings": [],
                                        "comments": [],
                                        "content_features": {}
                                    }
                                
                                feedback_data[feedback_type][age_group]["ratings"].append(feedback_item["rating"])
                                if feedback_item.get("comments"):
                                    feedback_data[feedback_type][age_group]["comments"].append(feedback_item["comments"])
                    except Exception as e:
                        logger.error(f"Error loading feedback file {feedback_file}: {str(e)}")
                        continue
                        
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return feedback_data

    @lru_cache(maxsize=1000)
    def _get_feedback_insights(self, age_group: int) -> Dict[str, float]:
        """Get insights from feedback for a specific age group with caching."""
        age_group_str = str(age_group)
        insights = {
            "age_appropriate": 0.0,
            "engagement": 0.0,
            "clarity": 0.0
        }
        
        # Calculate average ratings
        for feedback_type in ["age_appropriate", "engagement", "clarity"]:
            if age_group_str in self.feedback_data[feedback_type]:
                data = self.feedback_data[feedback_type][age_group_str]
                if data["ratings"]:
                    insights[feedback_type] = sum(data["ratings"]) / len(data["ratings"]) / 5.0
        
        return insights

    def calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """Calculate reward from feedback data."""
        weights = {
            'age_appropriate': 0.4,
            'engagement': 0.3,
            'clarity': 0.3
        }
        
        reward = 0.0
        for metric, weight in weights.items():
            if metric in feedback:
                # Normalize rating to [0, 1] range
                normalized_rating = (feedback[metric] - 1) / 4.0
                reward += normalized_rating * weight
        
        return reward

    def get_action(self, state: ContentState) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        """Train the Q-network using experience replay with optimized batch processing."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU
        states = torch.stack([s.to_tensor() for s in states]).to(self.device)
        next_states = torch.stack([s.to_tensor() for s in next_states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q-values with gradient scaling
        with torch.cuda.amp.autocast():
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize with gradient scaling
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network less frequently
        if self.steps % self.update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps += 1
        self.update_epsilon()
        
        return loss.item()

    def get_content_features(self, content: str) -> Dict[str, float]:
        """Extract features from content with caching."""
        # Generate cache key
        cache_key = hashlib.md5(content.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Calculate features
        words = content.split()
        sentences = content.split('.')
        
        features = {
            'vocabulary_complexity': len(set(words)) / len(words) if words else 0,
            'sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]),
            'interactive_elements': content.count('?') / len(sentences) if sentences else 0,
            'example_count': content.lower().count('for example') + content.lower().count('like'),
            'question_count': content.count('?')
        }
        
        # Cache results
        self._feature_cache[cache_key] = features
        return features

    def update_with_feedback(self, feedback_data: Dict[str, Any]):
        """Update the RL system with new feedback with optimized processing."""
        try:
            # Create state from feedback with caching
            state = ContentState(
                age_group=feedback_data['age_group'],
                category=feedback_data['category'],
                topic=feedback_data.get('topic', ''),
                content_features=self.get_content_features(feedback_data.get('content', ''))
            )
            
            # Calculate reward with caching
            cache_key = f"{feedback_data['age_group']}_{feedback_data['category']}_{hashlib.md5(str(feedback_data).encode()).hexdigest()}"
            if cache_key in self._feedback_cache:
                current_reward = self._feedback_cache[cache_key]
            else:
                current_reward = self.calculate_reward({
                    'age_appropriate': feedback_data.get('age_appropriate', 3),
                    'engagement': feedback_data.get('engagement', 3),
                    'clarity': feedback_data.get('clarity', 3)
                })
                self._feedback_cache[cache_key] = current_reward
            
            # Get previous reward if available
            previous_reward = 0.0
            if self.metrics_history:
                previous_entries = [entry for entry in self.metrics_history 
                                 if entry.get('type') == 'reward' 
                                 and entry.get('context', {}).get('age_group') == feedback_data['age_group']
                                 and entry.get('context', {}).get('category') == feedback_data['category']]
                if previous_entries:
                    previous_reward = previous_entries[-1].get('reward', 0.0)
            
            # Calculate improvement
            improvement = current_reward - previous_reward
            
            # Log improvement
            reinforcement_logger_instance.log_improvement(
                current_reward=current_reward,
                previous_reward=previous_reward,
                improvement=improvement,
                iteration=self.steps
            )
            
            # Get next state
            next_state = state
            
            # Add to replay buffer
            self.replay_buffer.add(state, 0, current_reward, next_state, True)
            
            # Train the network
            loss = self.train()
            
            # Log example with enhanced context
            if feedback_data.get('content'):
                example = Example(
                    topic=feedback_data.get('topic', ''),
                    age_group=feedback_data['age_group'],
                    context={
                        'category': feedback_data['category'],
                        'original_article': feedback_data.get('original_article', ''),
                        'ratings': {
                            'age_appropriate': feedback_data.get('age_appropriate', 3),
                            'engagement': feedback_data.get('engagement', 3),
                            'clarity': feedback_data.get('clarity', 3)
                        },
                        'comments': feedback_data.get('comments', '')
                    },
                    original_article=feedback_data.get('original_article', ''),
                    kid_friendly_text=feedback_data.get('content', ''),
                    age_appropriateness=feedback_data.get('age_appropriate', 3),
                    engagement=feedback_data.get('engagement', 3)
                )
                reinforcement_logger_instance.log_example(
                    example=example,
                    impact={
                        'reward': current_reward,
                        'loss': loss if loss is not None else 0.0,
                        'improvement': improvement
                    }
                )
            
        except Exception as e:
            logger.error(f"Error updating RL system: {str(e)}")

    def get_generation_guidelines(self, age_group: int, category: str) -> Dict[str, Any]:
        """Get content generation guidelines based on learned policy."""
        try:
            # Create a sample state
            state = ContentState(
                age_group=age_group,
                category=category,
                topic='',
                content_features={
                    'vocabulary_complexity': 0.5,
                    'sentence_length': 10,
                    'interactive_elements': 0.2,
                    'example_count': 1,
                    'question_count': 1
                }
            )
            
            # Get action from policy
            action = self.get_action(state)
            
            # Convert action to guidelines
            guidelines = {
                'vocabulary_complexity': max(0.1, min(1.0, 0.5 + (action - 5) * 0.1)),
                'sentence_length': max(5, min(20, 10 + (action - 5) * 2)),
                'interactive_elements': action > 5,
                'example_count': max(0, min(5, 1 + (action - 5))),
                'question_count': max(0, min(3, 1 + (action - 5) // 2))
            }
            
            return guidelines
            
        except Exception as e:
            logger.error(f"Error getting generation guidelines: {str(e)}")
            return {
                'vocabulary_complexity': 0.5,
                'sentence_length': 10,
                'interactive_elements': True,
                'example_count': 1,
                'question_count': 1
            }

    def save_model(self, path: str):
        """Save the Q-network model."""
        try:
            # Save Q-network
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, path)
            
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """Load the Q-network model."""
        try:
            # Load Q-network
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}") 