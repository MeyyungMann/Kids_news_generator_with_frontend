from pathlib import Path
from typing import List, Optional

class DirectoryManager:
    """Centralized directory management for the application."""
    
    def __init__(self):
        """Initialize directory paths."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.src_dir = self.base_dir / "src"
        self.results_dir = self.base_dir / "results"
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.offload_dir = self.base_dir / "offload"
        
        # Subdirectories
        self.images_dir = self.results_dir / "images"
        self.summaries_dir = self.results_dir / "summaries"
        self.feedback_dir = self.data_dir / "feedback"
        self.rag_documents_dir = self.data_dir / "rag_documents"
        self.rag_index_dir = self.data_dir / "rag_index"
    
    def create_directories(self, categories: Optional[List[str]] = None):
        """
        Create all necessary directories for the application.
        
        Args:
            categories: Optional list of categories to create subdirectories for
        """
        # Create main directories
        directories = [
            self.results_dir,
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.offload_dir,
            self.images_dir,
            self.summaries_dir,
            self.feedback_dir,
            self.rag_documents_dir,
            self.rag_index_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories if categories are provided
        if categories:
            for category in categories:
                (self.images_dir / category).mkdir(parents=True, exist_ok=True)
                (self.summaries_dir / category).mkdir(parents=True, exist_ok=True)
    
    def get_path(self, path_type: str) -> Path:
        """
        Get a specific directory path.
        
        Args:
            path_type: Type of directory to get path for
            
        Returns:
            Path object for the requested directory
        """
        path_map = {
            'base': self.base_dir,
            'src': self.src_dir,
            'results': self.results_dir,
            'models': self.models_dir,
            'data': self.data_dir,
            'logs': self.logs_dir,
            'offload': self.offload_dir,
            'images': self.images_dir,
            'summaries': self.summaries_dir,
            'feedback': self.feedback_dir,
            'rag_documents': self.rag_documents_dir,
            'rag_index': self.rag_index_dir
        }
        
        if path_type not in path_map:
            raise ValueError(f"Unknown path type: {path_type}")
        
        return path_map[path_type] 