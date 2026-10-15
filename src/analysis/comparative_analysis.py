"""
Comparative analysis of human vs AI voice features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ComparativeAnalyzer:
    """Analyze differences between human and AI voice features"""

    def __init__(self):
        """Initialize analyzer"""
        self.human_features = []
        self.ai_features = []

    def add_human_features(self, features: Dict[str, float]):
        """
        Add features from human voice sample

        Args:
            features: Dictionary of feature statistics
        """
        self.human_features.append(features)

    def add_ai_features(self, features: Dict[str, float]):
        """
        Add features from AI voice sample

        Args:
            features: Dictionary of feature statistics
        """
        self.ai_features.append(features)

    def create_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create pandas DataFrames from collected features

        Returns:
            Tuple of (human_df, ai_df)
        """
        human_df = pd.DataFrame(self.human_features)
        ai_df = pd.DataFrame(self.ai_features)
        return human_df, ai_df

    def statistical_test(self, feature_name: str, test: str = 'ttest') -> Dict[str, float]:
        """
        Perform statistical test on a specific feature

        Args:
            feature_name: Name of feature to test
            test: Type of test ('ttest', 'mannwhitney', 'ks')

        Returns:
            Dictionary with test statistic and p-value
        """
        human_df, ai_df = self.create_dataframe()

        if feature_name not in human_df.columns or feature_name not in ai_df.columns:
            raise ValueError(f"Feature {feature_name} not found in data")

        human_values = human_df[feature_name].dropna()
        ai_values = ai_df[feature_name].dropna()

        if test == 'ttest':
            statistic, p_value = stats.ttest_ind(human_values, ai_values)
        elif test == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(human_values, ai_values)
        elif test == 'ks':
            statistic, p_value = stats.ks_2samp(human_values, ai_values)
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        Rank features by discriminative power

        Returns:
            DataFrame with features ranked by p-value
        """
        human_df, ai_df = self.create_dataframe()

        results = []
        for feature in human_df.columns:
            if feature in ai_df.columns:
                try:
                    result = self.statistical_test(feature, test='mannwhitney')
                    results.append({
                        'feature': feature,
                        'p_value': result['p_value'],
                        'statistic': result['statistic'],
                        'significant': result['significant']
                    })
                except:
                    continue

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')
        return results_df

    def plot_feature_distribution(self, feature_name: str, save_path: str = None):
        """
        Plot distribution of a feature for human vs AI

        Args:
            feature_name: Name of feature to plot
            save_path: Optional path to save figure
        """
        human_df, ai_df = self.create_dataframe()

        plt.figure(figsize=(10, 6))

        plt.hist(human_df[feature_name].dropna(), alpha=0.5, label='Human', bins=30)
        plt.hist(ai_df[feature_name].dropna(), alpha=0.5, label='AI', bins=30)

        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature_name}: Human vs AI')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, save_path: str = None):
        """
        Plot correlation matrix of features

        Args:
            save_path: Optional path to save figure
        """
        human_df, ai_df = self.create_dataframe()

        # Combine and add label
        human_df['label'] = 'human'
        ai_df['label'] = 'ai'
        combined_df = pd.concat([human_df, ai_df], ignore_index=True)

        # Select only numeric columns
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = combined_df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self) -> str:
        """
        Generate summary report of findings

        Returns:
            String containing summary report
        """
        importance_df = self.feature_importance()

        report = "AI Voice Detection - Comparative Analysis Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Total samples analyzed:\n"
        report += f"  Human voices: {len(self.human_features)}\n"
        report += f"  AI voices: {len(self.ai_features)}\n\n"

        report += "Top discriminative features:\n"
        for idx, row in importance_df.head(10).iterrows():
            sig = "*" if row['significant'] else ""
            report += f"  {sig} {row['feature']}: p={row['p_value']:.2e}\n"

        report += "\n* indicates statistically significant difference (p < 0.05)\n"

        return report
