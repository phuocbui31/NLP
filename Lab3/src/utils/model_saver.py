import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from datetime import datetime

from loguru import logger

class ModelSaver:
    def __init__(self, base_path: str = "results"):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models"
        self.results_dir = self.base_path / "results"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.add(self.base_path / "logs/experiment.log",
                  rotation="1 day",
                  format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> Path:
        """
        Save a trained model with metadata.
        
        Args:
            model: The model to save
            model_name (str): Name for the saved model
            metadata (Dict): Additional metadata about the model
            
        Returns:
            Path: Path where the model was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = self.models_dir / filename
        
        # Save model
        with filepath.open('wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            metadata_file = filepath.with_suffix('.json')
            with metadata_file.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filepath: Path) -> Any:
        """
        Load a saved model.
        
        Args:
            filepath (Path): Path to the saved model
            
        Returns:
            Any: The loaded model
        """
        with Path(filepath).open('rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {filepath}")
        return model
    
    def save_results(self, results: Dict, experiment_name: str) -> Path:
        """
        Save experiment results.
        
        Args:
            results (Dict): Results dictionary to save
            experiment_name (str): Name of the experiment
            
        Returns:
            Path: Path where results were saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved: {filepath}")
        return filepath
    
    def save_similarity_results(self, word: str, similar_words: List[Tuple[str, float]], 
                              model_name: str) -> Path:
        """
        Save word similarity results.
        
        Args:
            word (str): Target word
            similar_words (List[Tuple[str, float]]): List of similar words with scores
            model_name (str): Name of the model used
            
        Returns:
            Path: Path where results were saved
        """
        results = {
            "target_word": word,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "similar_words": similar_words
        }
        
        filename = f"similarity_{word}_{model_name}"
        return self.save_results(results, filename)
    
    def save_vector_data(self, word: str, vector: np.ndarray, model_name: str) -> Path:
        """
        Save word vector data.
        
        Args:
            word (str): The word
            vector (np.ndarray): Word vector
            model_name (str): Name of the model
            
        Returns:
            Path: Path where vector was saved
        """
        results = {
            "word": word,
            "vector": vector.tolist(),
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "vector_size": len(vector)
        }
        
        filename = f"vector_{word}_{model_name}"
        return self.save_results(results, filename)
    
    def save_log(self, message: str, log_type: str = "info") -> None:
        """
        Save log message using loguru.
        
        Args:
            message (str): Log message
            log_type (str): Type of log (info, error, warning)
        """
        log_func = getattr(logger, log_type, logger.info)
        log_func(message)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def list_saved_models(self) -> List[Path]:
        """List all saved models."""
        return sorted(list(self.models_dir.glob("*.pkl")))
    
    def list_saved_results(self) -> List[Path]:
        """List all saved results."""
        return sorted(list(self.results_dir.glob("*.json")))
