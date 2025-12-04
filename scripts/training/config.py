"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model
    base_model: str = "roberta-base"  # Temporarily using RoBERTa (DeBERTa has tokenizer issues on Modal)
    # TODO: Switch back to "microsoft/deberta-v3-base" after fixing tokenizer
    # Alternative: "microsoft/deberta-v3-small" (smaller/faster)
    output_dir: str = "/vol/models/corpus-trained-roberta"

    # Data
    data_dir: str = "scripts/training/data"
    train_file: str = "train.jsonl"
    dev_file: str = "dev.jsonl"
    test_file: str = "test.jsonl"

    # Training hyperparameters
    learning_rate: float = 3e-5  # 3e-5 for RoBERTa, 2e-5 for DeBERTa-v3
    batch_size: int = 16
    num_epochs: int = 10
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Early stopping
    early_stopping_patience: int = 5  # Increased to allow more time for learning
    early_stopping_threshold: float = (
        0.01  # Increased threshold (0.01 point improvement)
    )

    # Evaluation
    eval_steps: int = 100  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    logging_steps: int = 50

    # Test run settings (for quick validation)
    test_run_max_samples: int = 100  # Limit samples for test runs
    test_run_max_steps: int = 50  # Limit steps for test runs

    # Output (IELTS-aligned CEFR mapping: A1=2.0 to C2=8.5)
    target_score_min: float = 2.0
    target_score_max: float = 8.5

    # Ordinal Regression Options
    use_ordinal_regression: bool = False  # Temporarily disabled to test baseline
    # TODO: Re-enable after verifying baseline works
    num_classes: int = (
        11  # 11 CEFR levels: A1, A1+, A2, A2+, B1, B1+, B2, B2+, C1, C1+, C2
    )

    # Loss Function
    loss_type: str = (
        "coral"  # Options: "mse" (baseline), "coral", "soft_labels", "focal", "cdw_ce"
    )
    soft_label_sigma: float = 1.0  # For soft_labels loss
    focal_alpha: float = 0.25  # For focal loss
    focal_gamma: float = 2.0  # For focal loss

    # Class Imbalance Handling
    use_class_weights: bool = False  # Enable class weighting in loss
    class_weight_power: float = 0.5  # Weight = 1 / (class_freq ** power)

    # Data Augmentation (for minority classes)
    augment_minority_classes: bool = False  # Enable data augmentation
    augmentation_factor: int = 2  # Oversample minority classes by this factor
    min_class_samples: int = 100  # Classes below this threshold get augmented

    def get_train_path(self) -> Path:
        """Get path to training data file."""
        return Path(self.data_dir) / self.train_file

    def get_dev_path(self) -> Path:
        """Get path to dev data file."""
        return Path(self.data_dir) / self.dev_file

    def get_test_path(self) -> Path:
        """Get path to test data file."""
        return Path(self.data_dir) / self.test_file


# Default config
DEFAULT_CONFIG = TrainingConfig()
