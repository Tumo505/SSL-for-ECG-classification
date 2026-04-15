#!/usr/bin/env python3
"""Statistical significance testing for model comparisons."""

import json
from pathlib import Path
import numpy as np
import argparse
from scipy import stats
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical significance testing.")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing evaluation results")
    parser.add_argument("--baseline", type=str, default="supervised",
                       choices=["supervised", "ssl"],
                       help="Baseline method to compare against")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/statistical_tests"),
                       help="Output directory for results")
    return parser.parse_args()


class StatisticalTester:
    """Perform statistical tests on model comparison results."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
    
    def paired_ttest(self, group1: np.ndarray, group2: np.ndarray, 
                     alternative: str = "two-sided") -> Dict[str, Any]:
        """Perform paired t-test.
        
        Args:
            group1: Metric values for group 1
            group2: Metric values for group 2
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Test results dictionary
        """
        t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
        
        # Cohen's d effect size
        diff = group1 - group2
        cohens_d = np.mean(diff) / np.std(diff)
        
        return {
            "test": "paired_ttest",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < self.alpha,
            "cohens_d": float(cohens_d),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "ci_lower": float(np.mean(diff) - 1.96 * np.std(diff) / np.sqrt(len(diff))),
            "ci_upper": float(np.mean(diff) + 1.96 * np.std(diff) / np.sqrt(len(diff))),
        }
    
    def mannwhitneyu_test(self, group1: np.ndarray, group2: np.ndarray,
                         alternative: str = "two-sided") -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric).
        
        Args:
            group1: Metric values for group 1
            group2: Metric values for group 2
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Test results dictionary
        """
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        # Rank-biserial correlation effect size
        r = 1 - (2 * u_stat) / (len(group1) * len(group2))
        
        return {
            "test": "mannwhitneyu",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < self.alpha,
            "rank_biserial_r": float(r),
            "median_group1": float(np.median(group1)),
            "median_group2": float(np.median(group2)),
            "median_diff": float(np.median(group1) - np.median(group2)),
        }
    
    def test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test if data is normally distributed using Shapiro-Wilk test."""
        if len(data) < 3:
            return {"test": "shapiro", "message": "Sample size too small"}
        
        stat, p_value = stats.shapiro(data)
        return {
            "test": "shapiro",
            "statistic": float(stat),
            "p_value": float(p_value),
            "normal": float(p_value) > self.alpha,
        }
    
    def compare_methods(self, method1_scores: Dict[str, List[float]],
                       method2_scores: Dict[str, List[float]],
                       method1_name: str = "Method 1",
                       method2_name: str = "Method 2") -> Dict[str, Any]:
        """Compare two methods across multiple metrics and seeds.
        
        Args:
            method1_scores: Dict mapping metric names to lists of scores
            method2_scores: Dict mapping metric names to lists of scores
            method1_name: Name of method 1
            method2_name: Name of method 2
        
        Returns:
            Comprehensive comparison results
        """
        comparison = {}
        
        for metric in method1_scores:
            if metric not in method2_scores:
                continue
            
            scores1 = np.array(method1_scores[metric])
            scores2 = np.array(method2_scores[metric])
            
            # Check normality
            norm1 = self.test_normality(scores1)
            norm2 = self.test_normality(scores2)
            
            # Select appropriate test
            if norm1["normal"] and norm2["normal"]:
                test_result = self.paired_ttest(scores1, scores2, alternative="two-sided")
            else:
                test_result = self.mannwhitneyu_test(scores1, scores2, alternative="two-sided")
            
            comparison[metric] = {
                "method1": {
                    "name": method1_name,
                    "mean": float(np.mean(scores1)),
                    "std": float(np.std(scores1)),
                    "median": float(np.median(scores1)),
                    "min": float(np.min(scores1)),
                    "max": float(np.max(scores1)),
                },
                "method2": {
                    "name": method2_name,
                    "mean": float(np.mean(scores2)),
                    "std": float(np.std(scores2)),
                    "median": float(np.median(scores2)),
                    "min": float(np.min(scores2)),
                    "max": float(np.max(scores2)),
                },
                "normality": {
                    "method1": norm1,
                    "method2": norm2,
                },
                "test_result": test_result,
                "improvement": float(np.mean(scores1) - np.mean(scores2)),
                "improvement_pct": float((np.mean(scores1) - np.mean(scores2)) / np.mean(scores2) * 100),
            }
        
        return comparison


def create_comparison_table(comparison: Dict[str, Any], output_path: Path) -> None:
    """Create and save a comparison table."""
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("STATISTICAL SIGNIFICANCE TEST RESULTS")
    lines.append("="*100 + "\n")
    
    for metric, results in comparison.items():
        method1 = results["method1"]
        method2 = results["method2"]
        test_result = results["test_result"]
        
        lines.append(f"\n[{metric}]")
        lines.append(f"  {method1['name']:20s}: {method1['mean']:.4f} ± {method1['std']:.4f}")
        lines.append(f"  {method2['name']:20s}: {method2['mean']:.4f} ± {method2['std']:.4f}")
        lines.append(f"  Improvement:              {results['improvement']:.4f} ({results['improvement_pct']:+.2f}%)")
        lines.append(f"  Test:                     {test_result['test']}")
        lines.append(f"  P-value:                  {test_result['p_value']:.4f} {'***' if test_result['p_value'] < 0.001 else '**' if test_result['p_value'] < 0.01 else '*' if test_result['p_value'] < 0.05 else 'ns'}")
        lines.append(f"  Significant:              {'Yes' if test_result['significant'] else 'No'}")
        
        if "cohens_d" in test_result:
            lines.append(f"  Effect size (Cohen's d):  {test_result['cohens_d']:.4f}")
    
    lines.append("\n" + "="*100 + "\n")
    
    # Print and save
    text = "\n".join(lines)
    print(text)
    
    with open(output_path, "w") as f:
        f.write(text)


def create_comparison_plots(comparison: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for comparisons."""
    
    metrics = list(comparison.keys())
    improvements = [comparison[m]["improvement_pct"] for m in metrics]
    p_values = [comparison[m]["test_result"]["p_value"] for m in metrics]
    
    # Improvement plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax1.barh(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Method Comparison: Performance Improvement', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(improvements):
        ax1.text(v, i, f' {v:+.2f}%', va='center', fontweight='bold')
    
    # P-value plot with significance threshold
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    ax2.barh(metrics, p_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax2.set_xlabel('P-value', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(axis='x', alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "significance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'significance_comparison.png'}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[STATISTICAL SIGNIFICANCE TESTING]")
    print(f"Results directory: {args.results_dir}")
    print(f"Baseline: {args.baseline}")
    print(f"Significance level: {args.alpha}")
    
    # TODO: Load results from experiments and perform tests
    # For now, this is a template
    
    tester = StatisticalTester(alpha=args.alpha)
    
    print("\n[Note] To run actual tests:")
    print("  1. Complete multi-seed experiments using run_multiple_seeds.py")
    print("  2. Evaluate each checkpoint using evaluate_all.py or similar")
    print("  3. Run this script with --results-dir pointing to saved metrics")
    
    print(f"\nTemplate output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
