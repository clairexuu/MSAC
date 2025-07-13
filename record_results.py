"""
Results recording and analysis script for CommonsenseEnhancedGraphSmile
Handles result collection, visualization, and comparison across experiments
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_training_curves, plot_confusion_matrix

class ResultsRecorder:
    """Record and analyze experimental results"""
    
    def __init__(self, results_base_dir="results"):
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        
        self.master_results_file = self.results_base_dir / "master_results.json"
        self.comparison_dir = self.results_base_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Load existing master results
        self.master_results = self._load_master_results()
    
    def _load_master_results(self):
        """Load master results file"""
        if self.master_results_file.exists():
            with open(self.master_results_file, 'r') as f:
                return json.load(f)
        return {
            "experiments": {},
            "summary": {
                "total_experiments": 0,
                "best_f1": 0.0,
                "best_experiment": None,
                "last_updated": None
            }
        }
    
    def _save_master_results(self):
        """Save master results file"""
        self.master_results["summary"]["last_updated"] = datetime.now().isoformat()
        with open(self.master_results_file, 'w') as f:
            json.dump(self.master_results, f, indent=2)
    
    def record_experiment(self, experiment_dir):
        """Record results from an experiment directory"""
        exp_path = Path(experiment_dir)
        
        if not exp_path.exists():
            print(f"Experiment directory not found: {exp_path}")
            return
        
        # Load experiment data
        config_path = exp_path / "config.json"
        summary_path = exp_path / "summary.json"
        metrics_path = exp_path / "metrics_history.pkl"
        predictions_path = exp_path / "best_predictions.pkl"
        
        if not all(p.exists() for p in [config_path, summary_path, metrics_path]):
            print(f"Missing required files in {exp_path}")
            return
        
        # Load data
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        with open(metrics_path, 'rb') as f:
            metrics_history = pickle.load(f)
        
        predictions = None
        if predictions_path.exists():
            with open(predictions_path, 'rb') as f:
                predictions = pickle.load(f)
        
        # Extract key information
        exp_name = summary.get('experiment_name', exp_path.name)
        timestamp = datetime.now().isoformat()
        
        # Compute additional metrics
        test_metrics = self._extract_final_metrics(metrics_history)
        
        # Record in master results
        experiment_record = {
            "name": exp_name,
            "timestamp": timestamp,
            "directory": str(exp_path),
            "config": config,
            "summary": summary,
            "final_metrics": test_metrics,
            "best_f1": summary.get('best_f1', 0.0),
            "best_epoch": summary.get('best_epoch', 0),
            "total_epochs": len(metrics_history.get('train_loss', [])),
            "converged": self._check_convergence(metrics_history),
            "stable": self._check_stability(metrics_history)
        }
        
        # Add to master results
        self.master_results["experiments"][exp_name] = experiment_record
        self.master_results["summary"]["total_experiments"] = len(self.master_results["experiments"])
        
        # Update best experiment
        if experiment_record["best_f1"] > self.master_results["summary"]["best_f1"]:
            self.master_results["summary"]["best_f1"] = experiment_record["best_f1"]
            self.master_results["summary"]["best_experiment"] = exp_name
        
        self._save_master_results()
        
        # Generate individual experiment report
        self._generate_experiment_report(experiment_record, metrics_history, predictions)
        
        print(f"Recorded experiment: {exp_name}")
        print(f"  Best F1: {experiment_record['best_f1']:.2f} at epoch {experiment_record['best_epoch']}")
        print(f"  Total epochs: {experiment_record['total_epochs']}")
        print(f"  Converged: {experiment_record['converged']}")
        print(f"  Stable: {experiment_record['stable']}")
    
    def _extract_final_metrics(self, metrics_history):
        """Extract final test metrics"""
        final_metrics = {}
        
        for key, values in metrics_history.items():
            if key.startswith('test_') and values:
                metric_name = key.replace('test_', '')
                final_metrics[metric_name] = {
                    'final': values[-1],
                    'best': max(values) if 'acc' in key or 'f1' in key else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return final_metrics
    
    def _check_convergence(self, metrics_history, window=10):
        """Check if training converged"""
        train_loss = metrics_history.get('train_loss', [])
        if len(train_loss) < window * 2:
            return False
        
        # Check if loss has stabilized in the last window epochs
        recent_loss = train_loss[-window:]
        early_loss = train_loss[-window*2:-window]
        
        recent_mean = np.mean(recent_loss)
        early_mean = np.mean(early_loss)
        
        # Converged if recent loss is not significantly different from earlier
        improvement = (early_mean - recent_mean) / early_mean
        return improvement < 0.01  # Less than 1% improvement
    
    def _check_stability(self, metrics_history, window=10):
        """Check if training is stable (not overfitting)"""
        train_f1 = metrics_history.get('train_emotion_f1', [])
        valid_f1 = metrics_history.get('valid_emotion_f1', [])
        
        if len(train_f1) < window or len(valid_f1) < window:
            return True
        
        # Check if validation F1 is following training F1
        recent_train = np.mean(train_f1[-window:])
        recent_valid = np.mean(valid_f1[-window:])
        
        gap = recent_train - recent_valid
        return gap < 10.0  # Less than 10% gap indicates stability
    
    def _generate_experiment_report(self, experiment, metrics_history, predictions):
        """Generate detailed report for single experiment"""
        exp_name = experiment['name']
        report_dir = self.results_base_dir / f"{exp_name}_report"
        report_dir.mkdir(exist_ok=True)
        
        # Training curves
        curves_path = report_dir / "training_curves.png"
        plot_training_curves(metrics_history, curves_path)
        
        # Confusion matrix and classification report
        if predictions is not None:
            preds_emo, labels_emo, preds_sen, labels_sen = predictions
            
            if preds_emo is not None and labels_emo is not None:
                # Confusion matrix
                emotion_classes = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
                cm_path = report_dir / "confusion_matrix.png"
                plot_confusion_matrix(labels_emo, preds_emo, emotion_classes, cm_path)
                
                # Classification report
                report_text = classification_report(labels_emo, preds_emo, 
                                                  target_names=emotion_classes, 
                                                  digits=4, zero_division=0)
                
                report_path = report_dir / "classification_report.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Experiment: {exp_name}\n")
                    f.write(f"Best F1: {experiment['best_f1']:.4f}\n")
                    f.write(f"Best Epoch: {experiment['best_epoch']}\n")
                    f.write("=" * 50 + "\n")
                    f.write(report_text)
        
        # Experiment summary
        summary_path = report_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        print(f"  Generated report in: {report_dir}")
    
    def compare_experiments(self, experiment_names=None):
        """Compare multiple experiments"""
        if experiment_names is None:
            experiment_names = list(self.master_results["experiments"].keys())
        
        if len(experiment_names) < 2:
            print("Need at least 2 experiments for comparison")
            return
        
        # Create comparison dataframe
        comparison_data = []
        
        for exp_name in experiment_names:
            if exp_name not in self.master_results["experiments"]:
                print(f"Experiment not found: {exp_name}")
                continue
            
            exp = self.master_results["experiments"][exp_name]
            
            # Extract key metrics
            row = {
                'experiment': exp_name,
                'best_f1': exp['best_f1'],
                'best_epoch': exp['best_epoch'],
                'total_epochs': exp['total_epochs'],
                'converged': exp['converged'],
                'stable': exp['stable'],
                'hidden_dim': exp['config']['model']['hidden_dim'],
                'learning_rate': exp['config']['training']['learning_rate'],
                'batch_size': exp['config']['training']['batch_size'],
                'mode1': exp['config']['model']['mode1'],
                'norm': exp['config']['model']['norm'],
                'att2': exp['config']['model']['att2'],
                'listener_state': exp['config']['model']['listener_state'],
                'loss_type': exp['config']['training']['loss_type']
            }
            
            # Add final test metrics
            final_metrics = exp.get('final_metrics', {})
            for metric_name, metric_data in final_metrics.items():
                if isinstance(metric_data, dict):
                    row[f'final_{metric_name}'] = metric_data.get('final', 0)
                    row[f'best_{metric_name}'] = metric_data.get('best', 0)
            
            comparison_data.append(row)
        
        if not comparison_data:
            print("No valid experiments found for comparison")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.comparison_dir / f"comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate comparison plots
        self._plot_comparison(df, timestamp)
        
        print(f"\nComparison Results:")
        print(f"Best performing experiment: {df.loc[df['best_f1'].idxmax(), 'experiment']}")
        print(f"Best F1 Score: {df['best_f1'].max():.4f}")
        print(f"\nComparison saved to: {csv_path}")
        
        return df
    
    def _plot_comparison(self, df, timestamp):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment Comparison', fontsize=16)
        
        # F1 scores
        axes[0, 0].bar(range(len(df)), df['best_f1'], color='skyblue')
        axes[0, 0].set_title('Best F1 Scores')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['experiment'], rotation=45, ha='right')
        
        # Convergence epochs
        axes[0, 1].bar(range(len(df)), df['best_epoch'], color='lightcoral')
        axes[0, 1].set_title('Best Epoch')
        axes[0, 1].set_ylabel('Epoch')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['experiment'], rotation=45, ha='right')
        
        # Learning rate vs F1
        axes[1, 0].scatter(df['learning_rate'], df['best_f1'], s=100, alpha=0.7)
        axes[1, 0].set_title('Learning Rate vs F1 Score')
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Best F1 Score')
        axes[1, 0].set_xscale('log')
        
        # Hidden dim vs F1
        axes[1, 1].scatter(df['hidden_dim'], df['best_f1'], s=100, alpha=0.7)
        axes[1, 1].set_title('Hidden Dimension vs F1 Score')
        axes[1, 1].set_xlabel('Hidden Dimension')
        axes[1, 1].set_ylabel('Best F1 Score')
        
        plt.tight_layout()
        
        plot_path = self.comparison_dir / f"comparison_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {plot_path}")
    
    def generate_master_report(self):
        """Generate master report with all experiments"""
        report_path = self.results_base_dir / "master_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CommonsenseEnhancedGraphSmile - Master Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .good {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>CommonsenseEnhancedGraphSmile - Master Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Experiments: {self.master_results['summary']['total_experiments']}</li>
                <li>Best F1 Score: {self.master_results['summary']['best_f1']:.4f}</li>
                <li>Best Experiment: {self.master_results['summary']['best_experiment']}</li>
            </ul>
            
            <h2>Experiment Results</h2>
            <table>
                <tr>
                    <th>Experiment</th>
                    <th>Best F1</th>
                    <th>Best Epoch</th>
                    <th>Total Epochs</th>
                    <th>Converged</th>
                    <th>Stable</th>
                    <th>Hidden Dim</th>
                    <th>Learning Rate</th>
                    <th>Mode1</th>
                    <th>Norm</th>
                </tr>
        """
        
        for exp_name, exp_data in self.master_results["experiments"].items():
            f1_class = "best" if exp_data["best_f1"] > 65 else "good" if exp_data["best_f1"] > 60 else "poor"
            
            html_content += f"""
                <tr class="{f1_class}">
                    <td>{exp_name}</td>
                    <td>{exp_data['best_f1']:.2f}</td>
                    <td>{exp_data['best_epoch']}</td>
                    <td>{exp_data['total_epochs']}</td>
                    <td>{'✓' if exp_data['converged'] else '✗'}</td>
                    <td>{'✓' if exp_data['stable'] else '✗'}</td>
                    <td>{exp_data['config']['model']['hidden_dim']}</td>
                    <td>{exp_data['config']['training']['learning_rate']}</td>
                    <td>{exp_data['config']['model']['mode1']}</td>
                    <td>{exp_data['config']['model']['norm']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Analysis</h2>
            <p>Performance thresholds:</p>
            <ul>
                <li><span class="best">Best</span>: F1 > 65%</li>
                <li><span class="good">Good</span>: 60% < F1 ≤ 65%</li>
                <li><span class="poor">Poor</span>: F1 ≤ 60%</li>
            </ul>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Master report generated: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Record and analyze experimental results')
    parser.add_argument('--record', type=str, help='Record experiment from directory')
    parser.add_argument('--compare', nargs='*', help='Compare specific experiments')
    parser.add_argument('--report', action='store_true', help='Generate master report')
    parser.add_argument('--results_dir', default='results', help='Results base directory')
    
    args = parser.parse_args()
    
    recorder = ResultsRecorder(args.results_dir)
    
    if args.record:
        recorder.record_experiment(args.record)
    
    if args.compare is not None:
        df = recorder.compare_experiments(args.compare if args.compare else None)
        if df is not None:
            print("\nComparison Summary:")
            print(df[['experiment', 'best_f1', 'best_epoch', 'converged', 'stable']].to_string(index=False))
    
    if args.report:
        recorder.generate_master_report()
    
    if not any([args.record, args.compare is not None, args.report]):
        print("Usage examples:")
        print("  Record experiment: python record_results.py --record results/experiment_name")
        print("  Compare all: python record_results.py --compare")
        print("  Compare specific: python record_results.py --compare exp1 exp2 exp3")
        print("  Generate report: python record_results.py --report")

if __name__ == '__main__':
    main()