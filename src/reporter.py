import os

class Reporter:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def generate_final_report(self, report_data):
        """
        report_data is a dict containing:
        - dataset_summary: dict
        - label_distribution: dict
        - split_info: dict
        - faci_stats: dict
        - policy_stats: dict
        - optimizer_summary: dict
        - final_metrics: dict
        - per_class: dict
        - discussion: str
        - limitations: str
        - future_work: str
        """
        
        md = f"""# Final Report: Nature-Inspired Augmentation Selection

## 1. Dataset Summary
- **Total Samples processed**: {report_data.get('dataset_summary', {}).get('total', 0)}
- **Classes**: {', '.join(report_data.get('dataset_summary', {}).get('classes', []))}

## 2. Label Distribution
"""
        for k, v in report_data.get('label_distribution', {}).items():
            md += f"- **{k}**: {v}\n"
            
        md += f"""
## 3. Train/Test Split
- **Training Set**: {report_data.get('split_info', {}).get('train_size', 0)}
- **Validation Set**: {report_data.get('split_info', {}).get('val_size', 0)}
- **Test Set**: {report_data.get('split_info', {}).get('test_size', 0)}

## 4. FACI Statistics
- **Average FACI Scalar**: {report_data.get('faci_stats', {}).get('avg_scalar', 0):.4f}
- **Average Complexity**: {report_data.get('faci_stats', {}).get('avg_complexity', 0):.4f}
- **Average Semantic Entropy**: {report_data.get('faci_stats', {}).get('avg_entropy', 0):.4f}

## 5. Policy Statistics
- **Total Policies Generated**: {report_data.get('policy_stats', {}).get('total', 0)}
"""
        for k, v in report_data.get('policy_stats', {}).get('strategies', {}).items():
            md += f"- **{k}**: {v}\n"
            
        md += f"""
## 6. Optimizer Summary
- **Average Expected Utility**: {report_data.get('optimizer_summary', {}).get('avg_utility', 0):.4f}
- **Final Fitness**: {report_data.get('optimizer_summary', {}).get('final_fitness', 0):.4f}

## 7. Training Curves & Visualizations
*(See `visualizations/` folder for `training_loss.png`, `macro_f1.png`, `optimizer_convergence.png`, `faci_distribution.png`, `policy_distribution.png`, `replay_distribution.png`, `confusion_matrix.png`)*

## 8. Evaluation Metrics (Final Test Set)
- **Loss**: {report_data.get('final_metrics', {}).get('loss', 0):.4f}
- **Accuracy**: {report_data.get('final_metrics', {}).get('accuracy', 0):.4f}
- **Precision (Macro)**: {report_data.get('final_metrics', {}).get('precision', 0):.4f}
- **Recall (Macro)**: {report_data.get('final_metrics', {}).get('recall', 0):.4f}
- **Macro F1**: {report_data.get('final_metrics', {}).get('macro_f1', 0):.4f}
- **Weighted F1**: {report_data.get('final_metrics', {}).get('weighted_f1', 0):.4f}

## 9. Per-Class Performance
"""
        for k, v in report_data.get('per_class', {}).items():
            md += f"- **{k}**: F1 = {v:.4f}\n"
            
        md += f"""
## 10. Confusion Matrix
*(See `visualizations/confusion_matrix.png`)*

## 11. Discussion
{report_data.get('discussion', 'The Nature-Inspired Selection Framework successfully balanced augmentation budgets based on semantic complexity.')}

## 12. Limitations
{report_data.get('limitations', 'Dataset size was small, limiting deep learning baseline performance.')}

## 13. Future Work
{report_data.get('future_work', 'Integration of Butterfly Optimization Algorithm (BOA) and dynamic class imbalance weighting.')}
"""

        with open(os.path.join(self.results_dir, "final_report.md"), "w", encoding="utf-8") as f:
            f.write(md)
