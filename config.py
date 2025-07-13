"""
Configuration file for CommonsenseEnhancedGraphSmile model
Combines parameters from COSMIC and GraphSmile with adjustments for the hybrid model

ALL PARAMETERS ARE CONFIGURED HERE - NO COMMAND LINE ARGS SUPPORTED
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    # Model Architecture
    embedding_dims: List[int] = (1024, 342, 300)  # Text, Visual, Audio dimensions
    hidden_dim: int = 384  # GraphSmile hidden dimension
    n_classes: int = 7  # Number of emotion classes
    
    # COSMIC dimensions
    D_s: int = 768  # COMET feature dimension
    D_g: int = 150  # Global state dimension  
    D_p: int = 150  # Party state dimension
    D_r: int = 150  # Reaction state dimension
    D_i: int = 150  # Intent state dimension
    D_h: int = 100  # COSMIC hidden dimension
    D_a_att: int = 100  # Attention dimension
    
    # GraphSmile graph parameters (MELD configuration)
    heter_n_layers: List[int] = (5, 5, 5)  # Graph layers for TV, TA, VA
    win_p: int = 3  # Past window size
    win_f: int = 3  # Future window size
    shift_win: int = 3  # Sentiment shift window
    
    # COSMIC text processing
    mode1: int = 2  # RoBERTa fusion mode (0=concat all, 1=concat2, 2=avg4, etc.)
    norm: int = 0  # Normalization strategy (0=none, 1=LayerNorm, 3=BatchNorm)
    listener_state: bool = False  # Active listener
    context_attention: str = 'simple'  # Context attention type ('simple' or 'matching')
    emo_gru: bool = True  # Use GRU for emotion state
    att2: bool = True  # Use MatchingAttention for final classification
    residual: bool = False  # Residual connections
    
    # Dropout
    dropout: float = 0.2  # Standard dropout
    dropout_rec: float = 0.5  # Recurrent dropout

@dataclass  
class TrainingConfig:
    # Training parameters
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 7e-05  # GraphSmile's MELD learning rate
    l2_reg: float = 0.0001  # L2 regularization
    valid_ratio: float = 0.1  # Validation split ratio
    
    # Loss configuration
    loss_type: str = "emo_sen_sft"  # Loss combination type
    lambd: List[float] = (1.0, 0.5, 0.2)  # [emotion, sentiment, shift] weights
    class_weight: bool = True  # Use class weights for imbalanced data
    class_weight_mu: float = 0.0  # use predefined from COSMIC if 0, otherwise compute
    
    # Optimization
    optimizer: str = 'AdamW'  # Optimizer type
    scheduler: Optional[str] = None  # Learning rate scheduler
    gradient_clip: float = 1.0  # Gradient clipping
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001

@dataclass
class DataConfig:
    # Data paths
    data_path: str = "features/"
    roberta_path: str = "meld_features_roberta.pkl"
    comet_path: str = "meld_features_comet.pkl" 
    multimodal_path: str = "meld_multi_features.pkl"
    
    # Data processing
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class SystemConfig:
    # Hardware
    no_cuda: bool = False
    gpu_ids: str = "0"
    device: str = "cuda"
    
    # Distributed training  
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Reproducibility
    seed: int = 1234
    deterministic: bool = True
    
    # Logging
    tensorboard: bool = True
    log_interval: int = 10
    save_interval: int = 10

@dataclass
class ExperimentConfig:
    # Experiment identification
    name: str = "commonsense_enhanced_graphsmile"
    version: str = "v1.0"
    description: str = "COSMIC commonsense reasoning integrated with GraphSmile multimodal fusion"
    
    # Paths
    results_dir: str = "results"
    models_dir: str = "models" 
    logs_dir: str = "logs"
    
    # Saving
    save_best_only: bool = True
    save_last: bool = True
    save_top_k: int = 3

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        self.experiment = ExperimentConfig()

# Predefined configurations for different experiments
CONFIGS = {
    'baseline': {
        'model': {
            'mode1': 0,  # Concat all RoBERTa features
            'norm': 0,   # No normalization
            'att2': False,  # Simple classification
            'listener_state': False,
        },
        'training': {
            'learning_rate': 7e-05,
            'lambd': [1.0, 0.5, 0.2],
        }
    },
    
    'cosmic_enhanced': {
        'model': {
            'mode1': 2,  # Average RoBERTa features (COSMIC default)
            'norm': 1,   # LayerNorm
            'att2': True,  # Use MatchingAttention
            'listener_state': True,  # Active listener
            'context_attention': 'matching',
        },
        'training': {
            'learning_rate': 1e-04,  # COSMIC learning rate
            'lambd': [1.0, 0.5, 0.2],
        }
    },
    
    'graphsmile_focus': {
        'model': {
            'mode1': 0,  # Concat all features
            'norm': 0,   # No normalization  
            'att2': False,  # Simple classification
            'win_p': 17, 'win_f': 17,  # Large context windows
            'heter_n_layers': [6, 6, 6],
        },
        'training': {
            'learning_rate': 1e-05,  # Lower LR for larger windows
            'lambd': [1.0, 1.0, 1.0],  # Equal weights
        }
    }
}

def get_config(config_name='baseline'):
    """Get a predefined configuration"""
    config = Config()
    
    if config_name in CONFIGS:
        preset = CONFIGS[config_name]
        
        # Update model config
        if 'model' in preset:
            for key, value in preset['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Update training config
        if 'training' in preset:
            for key, value in preset['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
    
    return config

if __name__ == '__main__':
    # Example usage - NO COMMAND LINE ARGS
    config = Config()
    
    print("Model Config:")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  RoBERTa mode: {config.model.mode1}")
    print(f"  Graph layers: {config.model.heter_n_layers}")
    
    print("Training Config:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Class weights: {config.training.class_weight}")