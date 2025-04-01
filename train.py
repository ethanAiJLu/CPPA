# Standard libraries
import argparse

# PyTorch and related libraries
import torch
import torch.distributed as dist

# Dassl framework imports
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# Standard datasets
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

# -------------------------Custom datasets-------------------------

# Remote sensing datasets

"""
Each dataset module provides different data loading strategies:
- Standard modules: Source and target domain loaders with all samples
- ONE suffix: Single domain data
- TWO suffix: Few-shot domain adaptation with limited source domain labeled data
"""

# Standard domain adaptation (full sample mode)
import datasets.DARS_AID_UCM
import datasets.DARS_UCM_NWPU
import datasets.DARS_UCM_WHU
import datasets.DARS_AID_NWPU
import datasets.DARS_UCM_AID_WHU_RSSCN7

# Single domain training (single domain mode)
import datasets.DARS_AID_UCM_ONE
import datasets.DARS_UCM_NWPU_ONE
import datasets.DARS_UCM_WHU_ONE
import datasets.DARS_AID_NWPU_ONE
import datasets.DARS_UCM_AID_WHU_RSSCN7_ONE

# Few-shot domain adaptation (few-shot mode)
import datasets.DARS_AID_NWPU_TWO
import datasets.DARS_UCM_WHU_TWO
import datasets.DARS_AID_UCM_TWO
import datasets.DARS_UCM_NWPU_TWO
import datasets.DARS_AID_NWPU_TWO
import datasets.DARS_UCM_AID_WHU_RSSCN7_TWO

# Standard Domain adaptation datasets
import datasets.office31_CPPA
import datasets.office_home_CPPA
import datasets.miniDomainNet_CPPA
import datasets.imageclef_CPPA
import datasets.VLCS_CPPA
import datasets.DG5_CPPA
import datasets.miniDomainNetNumshots
import datasets.visda2017_CPPA

# -------------------------Custom datasets-------------------------

import trainers.cppa

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

def extend_cfg(cfg):
    """
    Add new configuration variables to the global config object.
    
    This function extends the configuration with model-specific parameters
    needed for the Cross-Modal Prompt-guided Prototype Alignment (CPPA) model.
    
    Args:
        cfg: The configuration object to extend
    """
    from yacs.config import CfgNode as CN
    
    # Configuration for CPPA (Cross-Modal Prompt-guided Prototype Alignment)
    cfg.TRAINER.CPPA = CN()
    
    # Prompt learning parameters
    cfg.TRAINER.CPPA.N_CTX = 2                # Number of context vectors
    cfg.TRAINER.CPPA.CTX_INIT = "a photo of a"  # Text used to initialize context tokens
    cfg.TRAINER.CPPA.PREC = "fp16"            # Precision: fp16, fp32, or amp
    
    # Model architecture parameters
    cfg.TRAINER.CPPA.PROMPT_DEPTH = 9         # Depth of prompt interaction (max 12)
                                             # For depth=1, behaves like shallow prompting
    
    # Class selection parameter
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"     # Class subset to use: "all", "base", or "new"
    
    # Advanced model configurations
    cfg.TRAINER.CPPA.FUSING = "mean"          # Feature fusion strategy ("mean", "max", "first")
    cfg.TRAINER.CPPA.PS = True                # Whether to use parameter sharing
    
    # Runtime configuration
    cfg.eval_only = args.eval_only            # Evaluation-only mode flag
    cfg.random_init = args.random_init        # Random initialization flag
    

def setup_cfg(args):
    """
    Set up configuration by merging config files and command-line arguments.
    
    This function handles the configuration setup process with the following priority:
    1. Base default configuration
    2. Dataset-specific configuration file
    3. Method-specific configuration file
    4. Command-line arguments (direct parameters)
    5. Optional command-line arguments (via opts)
    
    Args:
        args: Command-line arguments object
        
    Returns:
        cfg: The complete configuration object
    """
    # Get default configuration
    cfg = get_cfg_default()
    
    # Add model-specific configuration options
    extend_cfg(cfg)

    # Configuration merging process
    # 1. Load dataset-specific configuration
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. Load method-specific configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. Override with direct command-line arguments
    reset_cfg(cfg, args)

    # 4. Apply any remaining optional arguments
    cfg.merge_from_list(args.opts)

    # Note: We don't freeze the configuration to allow runtime modifications
    # cfg.freeze()

    return cfg

def main(args):
    """
    Main execution function that handles the training and evaluation workflow.
    """
    # Setup configuration and environment
    cfg = setup_cfg(args)
    
    # Set random seed if specified
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        
    # Initialize logging and configure CUDA
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    # Display configuration and system information
    print_args(args, cfg)
    print("Collecting environment information...")
    print("** System information **\n{}\n".format(collect_env_info()))
    
    # Initialize trainer
    trainer = build_trainer(cfg)
    
    # Run evaluation or training based on arguments
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
        
    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    """Command line argument parsing and main execution entry point."""
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Training and evaluation script for CPPA model"
    )
    
    # Dataset and output configuration
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for logs and checkpoints",
    )
    
    # Training continuation
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint directory from which to resume training",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (only positive values enable fixed seed)",
    )
    
    # Domain adaptation settings
    parser.add_argument(
        "--source-domains",
        type=str,
        nargs="+",
        help="Source domains for domain adaptation/generalization",
    )
    parser.add_argument(
        "--target-domains",
        type=str,
        nargs="+",
        help="Target domains for domain adaptation/generalization",
    )
    
    # Data processing
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        help="Data augmentation methods to apply",
    )
    
    # Configuration files
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Path to method configuration file",
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="Path to dataset configuration file",
    )
    
    # Model configuration
    parser.add_argument(
        "--trainer",
        type=str,
        default="",
        help="Name of trainer to use",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="",
        help="Name of CNN backbone architecture",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="",
        help="Name of classification head",
    )
    
    # Execution modes
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (no training)",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use random initialization for model parameters",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training phase",
    )
    
    # Model loading for evaluation
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Directory containing model to load for evaluation mode",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        help="Specific epoch to load for evaluation",
    )
    
    # Additional options
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Additional configuration options to modify via command-line",
    )
    
    # Parse arguments and execute main function
    args = parser.parse_args()
    main(args)