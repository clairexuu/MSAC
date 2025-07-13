"""
Experiment runner for CommonsenseEnhancedGraphSmile
Run multiple experiments with different configurations
"""

import os
import sys
import subprocess
import itertools
from pathlib import Path
from config import Config, CONFIGS, get_config

def run_experiment(config_name, custom_args=None, use_config_object=True):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [sys.executable, "train_commonsense_enhanced.py"]
    
    # Use config object if available
    if use_config_object and config_name in CONFIGS:
        # Get config object and pass paths if needed
        config = get_config(config_name)
        
        # Add data path argument if different from default
        if hasattr(config.data, 'data_path') and config.data.data_path != 'features/':
            cmd.extend(['--data_path', config.data.data_path])
    
    # Add configuration-specific arguments
    if config_name in CONFIGS:
        config_preset = CONFIGS[config_name]
        
        # Model arguments
        if 'model' in config_preset:
            model_config = config_preset['model']
            
            if 'mode1' in model_config:
                cmd.extend(['--mode1', str(model_config['mode1'])])
            if 'norm' in model_config:
                cmd.extend(['--norm', str(model_config['norm'])])
            if 'att2' in model_config and model_config['att2']:
                cmd.append('--att2')
            if 'listener_state' in model_config and model_config['listener_state']:
                cmd.append('--active_listener')
            if 'context_attention' in model_config:
                cmd.extend(['--attention', model_config['context_attention']])
            if 'residual' in model_config and model_config['residual']:
                cmd.append('--residual')
            if 'win_p' in model_config and 'win_f' in model_config:
                cmd.extend(['--win', str(model_config['win_p']), str(model_config['win_f'])])
            if 'heter_n_layers' in model_config:
                cmd.extend(['--heter_n_layers'] + [str(x) for x in model_config['heter_n_layers']])
        
        # Training arguments
        if 'training' in config_preset:
            training_config = config_preset['training']
            
            if 'learning_rate' in training_config:
                cmd.extend(['--lr', str(training_config['learning_rate'])])
            if 'lambd' in training_config:
                cmd.extend(['--lambd'] + [str(x) for x in training_config['lambd']])
    
    # Add experiment name
    cmd.extend(['--name', f'{config_name}'])
    
    # Add custom arguments
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Experiment {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment {config_name} failed with error: {e}")
        return False

def run_grid_search(config_base='baseline'):
    """Run grid search over hyperparameters"""
    print("Starting Grid Search...")
    
    # Get base configuration
    base_config = get_config(config_base)
    
    # Define parameter grid based on config structure
    param_grid = {
        'hidden_dim': [256, 384, 512],
        'learning_rate': [1e-05, 7e-05, 1e-04],
        'mode1': [0, 1, 2],
        'norm': [0, 1],
        'batch_size': [8, 16, 32]
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Total combinations: {len(combinations)}")
    
    successful_runs = 0
    failed_runs = 0
    
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        
        exp_name = f"grid_search_{i+1:03d}"
        custom_args = [
            '--name', exp_name,
            '--hidden_dim', str(params['hidden_dim']),
            '--lr', str(params['learning_rate']),
            '--mode1', str(params['mode1']),
            '--norm', str(params['norm']),
            '--batch_size', str(params['batch_size']),
            '--epochs', '30'  # Shorter for grid search
        ]
        
        print(f"\nGrid Search {i+1}/{len(combinations)}: {params}")
        
        success = run_experiment('baseline', custom_args)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
    
    print(f"\nGrid Search Complete!")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/len(combinations)*100:.1f}%")

def run_ablation_study(config_base='baseline'):
    """Run ablation study to understand component contributions"""
    print("Starting Ablation Study...")
    
    # Get base configuration
    base_config = get_config(config_base)
    
    # Base configuration arguments
    base_args = [
        '--epochs', str(base_config.training.epochs),
        '--batch_size', str(base_config.training.batch_size),
        '--lr', str(base_config.training.learning_rate),
        '--hidden_dim', str(base_config.model.hidden_dim)
    ]
    
    # Ablation experiments
    ablations = [
        {
            'name': 'ablation_no_commonsense',
            'description': 'GraphSmile only (no COSMIC)',
            'args': ['--mode1', '0', '--norm', '0', '--loss_type', 'emo_sen_sft']
        },
        {
            'name': 'ablation_cosmic_only', 
            'description': 'COSMIC only (no GraphSmile graphs)',
            'args': ['--mode1', '2', '--norm', '1', '--att2', '--loss_type', 'cosmic_only']
        },
        {
            'name': 'ablation_no_attention',
            'description': 'No MatchingAttention for classification',
            'args': ['--mode1', '2', '--norm', '1', '--loss_type', 'cosmic_graphsmile']
        },
        {
            'name': 'ablation_no_listener',
            'description': 'No active listener state',
            'args': ['--mode1', '2', '--norm', '1', '--att2', '--loss_type', 'cosmic_graphsmile']
        },
        {
            'name': 'ablation_full_model',
            'description': 'Full CommonsenseEnhancedGraphSmile',
            'args': ['--mode1', '2', '--norm', '1', '--att2', '--active_listener', '--loss_type', 'cosmic_graphsmile']
        }
    ]
    
    successful_runs = 0
    
    for ablation in ablations:
        print(f"\nRunning: {ablation['name']}")
        print(f"Description: {ablation['description']}")
        
        custom_args = base_args + ablation['args'] + ['--name', ablation['name']]
        
        success = run_experiment('baseline', custom_args)
        if success:
            successful_runs += 1
    
    print(f"\nAblation Study Complete!")
    print(f"Successful runs: {successful_runs}/{len(ablations)}")

def run_comparison_study(config_base='cosmic_enhanced'):
    """Run comparison with different loss strategies"""
    print("Starting Loss Strategy Comparison...")
    
    # Get base configuration
    base_config = get_config(config_base)
    
    base_args = [
        '--epochs', str(base_config.training.epochs),
        '--batch_size', str(base_config.training.batch_size), 
        '--lr', str(base_config.training.learning_rate),
        '--hidden_dim', str(base_config.model.hidden_dim),
        '--mode1', str(base_config.model.mode1),
        '--norm', str(base_config.model.norm)
    ]
    
    # Add conditional arguments
    if base_config.model.att2:
        base_args.append('--att2')
    if base_config.model.listener_state:
        base_args.append('--active_listener')
    
    loss_strategies = [
        {
            'name': 'loss_emo_only',
            'args': ['--loss_type', 'emo', '--lambd', '1.0']
        },
        {
            'name': 'loss_cosmic_only', 
            'args': ['--loss_type', 'cosmic_only']
        },
        {
            'name': 'loss_equal_weights',
            'args': ['--loss_type', 'emo_sen_sft', '--lambd', '1.0', '1.0', '1.0']
        },
        {
            'name': 'loss_graphsmile_weights',
            'args': ['--loss_type', 'emo_sen_sft', '--lambd', '1.0', '0.5', '0.2']
        },
        {
            'name': 'loss_combined',
            'args': ['--loss_type', 'cosmic_graphsmile', '--lambd', '1.0', '0.5', '0.2']
        },
        {
            'name': 'loss_epoch_weighted',
            'args': ['--loss_type', 'epoch_weighted', '--lambd', '1.0', '0.5', '0.2']
        }
    ]
    
    successful_runs = 0
    
    for strategy in loss_strategies:
        print(f"\nRunning: {strategy['name']}")
        
        custom_args = base_args + strategy['args'] + ['--name', strategy['name']]
        
        success = run_experiment('baseline', custom_args)
        if success:
            successful_runs += 1
    
    print(f"\nLoss Strategy Comparison Complete!")
    print(f"Successful runs: {successful_runs}/{len(loss_strategies)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CommonsenseEnhancedGraphSmile experiments')
    parser.add_argument('--mode', choices=['single', 'presets', 'grid', 'ablation', 'comparison'], 
                       default='presets', help='Experiment mode')
    parser.add_argument('--config', default='baseline', help='Configuration name for single mode or base config for studies')
    parser.add_argument('--custom_args', nargs='*', help='Custom arguments for single mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Run single experiment
        success = run_experiment(args.config, args.custom_args)
        if success:
            print(f"\n✓ Single experiment completed successfully")
        else:
            print(f"\n✗ Single experiment failed")
    
    elif args.mode == 'presets':
        # Run all preset configurations
        print("Running Preset Configurations...")
        
        successful_runs = 0
        total_runs = len(CONFIGS)
        
        for config_name in CONFIGS.keys():
            success = run_experiment(config_name, use_config_object=True)
            if success:
                successful_runs += 1
        
        print(f"\nPreset Experiments Complete!")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        print(f"Success rate: {successful_runs/total_runs*100:.1f}%")
    
    elif args.mode == 'grid':
        run_grid_search(args.config)
    
    elif args.mode == 'ablation':
        run_ablation_study(args.config)
    
    elif args.mode == 'comparison':
        run_comparison_study(args.config)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Use record_results.py to analyze and compare results.")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()