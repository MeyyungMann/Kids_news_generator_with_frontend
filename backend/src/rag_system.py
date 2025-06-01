import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_documents: int = 1000):
        """Initialize the RAG system with FAISS index and sentence transformer."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.max_documents = max_documents
        
        # Create directory for storing the index
        self.index_dir = Path(__file__).parent.parent / "data" / "rag_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_or_create_index()
        
        # Add news article handling
        self.news_documents_dir = Path(__file__).parent.parent / "data" / "rag_documents"
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load the index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.documents = metadata['documents']
                    self.document_metadata = metadata['document_metadata']
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading existing index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create a new index with the same dimension as the sentence transformer
        dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def _save_index(self):
        """Save the current index and metadata to disk."""
        try:
            # Save the index
            index_path = self.index_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.index_dir / "metadata.json"
            metadata = {
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'last_updated': datetime.now().isoformat()
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Saved index and metadata successfully")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def _maintain_sliding_window(self, new_documents: List[str], new_metadata: List[Dict[str, Any]]):
        """Maintain a sliding window of documents by removing oldest documents when limit is exceeded."""
        # Add new documents and metadata
        self.documents.extend(new_documents)
        self.document_metadata.extend(new_metadata)
        
        # If we exceed the limit, remove oldest documents
        if len(self.documents) > self.max_documents:
            # Calculate how many documents to remove
            num_to_remove = len(self.documents) - self.max_documents
            
            # Remove oldest documents and metadata
            self.documents = self.documents[num_to_remove:]
            self.document_metadata = self.document_metadata[num_to_remove:]
            
            logger.info(f"Removed {num_to_remove} oldest documents to maintain limit of {self.max_documents}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the index."""
        try:
            # Generate embeddings for new documents
            embeddings = self.model.encode(documents, show_progress_bar=True)
            
            # Maintain sliding window of documents
            self._maintain_sliding_window(documents, metadata or [{} for _ in documents])
            
            # Create new index with current documents
            self._create_or_update_index()
            
            logger.info(f"Added {len(documents)} documents to the index (total: {len(self.documents)})")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search the index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'),
                k
            )
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # Ensure index is valid
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'score': float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def validate_facts(self, text: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Validate facts in the generated text against the document store."""
        try:
            # Split text into sentences for fact validation
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            validation_results = {
                'validated_facts': [],
                'unverified_facts': [],
                'overall_score': 0.0
            }
            
            total_score = 0.0
            for sentence in sentences:
                # Search for relevant documents
                results = self.search(sentence, k=3)
                
                if results:
                    # Calculate similarity score
                    max_score = max(r['score'] for r in results)
                    total_score += max_score
                    
                    if max_score >= threshold:
                        validation_results['validated_facts'].append({
                            'fact': sentence,
                            'score': max_score,
                            'supporting_docs': results[:2]  # Keep top 2 supporting documents
                        })
                    else:
                        validation_results['unverified_facts'].append({
                            'fact': sentence,
                            'score': max_score
                        })
                else:
                    validation_results['unverified_facts'].append({
                        'fact': sentence,
                        'score': 0.0
                    })
            
            # Calculate overall validation score
            if sentences:
                validation_results['overall_score'] = total_score / len(sentences)
            
            return validation_results
        except Exception as e:
            logger.error(f"Error validating facts: {str(e)}")
            return {
                'validated_facts': [],
                'unverified_facts': [],
                'overall_score': 0.0,
                'error': str(e)
            }
    
    def load_news_articles(self):
        """Load news articles from the RAG documents directory."""
        try:
            if not self.news_documents_dir.exists():
                logger.warning("No news articles directory found")
                return
            
            # Clear existing documents to start fresh
            self.documents = []
            self.document_metadata = []
            
            # Load articles from each category
            for category_dir in self.news_documents_dir.iterdir():
                if category_dir.is_dir():
                    logger.info(f"Loading articles from category: {category_dir.name}")
                    
                    for article_file in category_dir.glob("*.json"):
                        try:
                            with open(article_file, 'r', encoding='utf-8') as f:
                                article = json.load(f)
                                
                                # Add each chunk as a separate document
                                for chunk in article['chunks']:
                                    self.documents.append(chunk)
                                    self.document_metadata.append({
                                        'title': article['title'],
                                        'source': article['source'],
                                        'url': article['url'],
                                        'category': article['category'],
                                        'published_at': article['published_at']
                                    })
                            
                            logger.info(f"Loaded article: {article_file.name}")
                        except Exception as e:
                            logger.error(f"Error loading article {article_file.name}: {str(e)}")
                            continue
            
            # Apply document limit if needed
            if len(self.documents) > self.max_documents:
                logger.info(f"Limiting documents from {len(self.documents)} to {self.max_documents}")
                # Keep only the most recent documents
                self.documents = self.documents[-self.max_documents:]
                self.document_metadata = self.document_metadata[-self.max_documents:]
            
            # Create or update the index
            if self.documents:
                self._create_or_update_index()
                logger.info(f"Loaded {len(self.documents)} document chunks from news articles")
            else:
                logger.warning("No document chunks found in news articles")
                
        except Exception as e:
            logger.error(f"Error loading news articles: {str(e)}")
    
    def _create_or_update_index(self):
        """Create or update the FAISS index with current documents."""
        try:
            # Generate embeddings for all documents
            embeddings = self.model.encode(self.documents, show_progress_bar=True)
            
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Save the index
            self._save_index()
            
            logger.info(f"Created/updated index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error creating/updating index: {str(e)}") 