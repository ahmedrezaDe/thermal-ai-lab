"""
Data Analysis and Visualization Module

This module provides tools for analyzing thermal system data,
including statistical analysis, visualization, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple
import json


class ThermalDataAnalyzer:
    """Analyzer for thermal system data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with data.
        
        Args:
            data: DataFrame containing thermal system data
        """
        self.data = data
        self.analysis_results = {}
        
    def basic_statistics(self) -> Dict:
        """
        Calculate basic statistical properties.
        
        Returns:
            Dictionary of statistics
        """
        stats_dict = {}
        for column in self.data.select_dtypes(include=[np.number]).columns:
            col_data = self.data[column].dropna()
            stats_dict[column] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data)
            }
        
        self.analysis_results['basic_statistics'] = stats_dict
        return stats_dict
    
    def time_series_analysis(self, time_column: str, value_column: str) -> Dict:
        """
        Analyze time series properties.
        
        Args:
            time_column: Name of time column
            value_column: Name of value column
            
        Returns:
            Dictionary of time series analysis results
        """
        ts_data = self.data[[time_column, value_column]].dropna()
        
        # Calculate time differences
        time_diffs = np.diff(ts_data[time_column].values)
        
        # Calculate rate of change
        value_diffs = np.diff(ts_data[value_column].values)
        rates = value_diffs / (time_diffs + 1e-8)
        
        results = {
            'duration': ts_data[time_column].max() - ts_data[time_column].min(),
            'mean_rate': np.mean(rates),
            'max_rate': np.max(rates),
            'min_rate': np.min(rates),
            'std_rate': np.std(rates),
            'total_change': ts_data[value_column].iloc[-1] - ts_data[value_column].iloc[0]
        }
        
        self.analysis_results['time_series'] = results
        return results
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Returns:
            Correlation DataFrame
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        self.analysis_results['correlation'] = correlation_matrix
        return correlation_matrix
    
    def detect_outliers(self, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in specified column.
        
        Args:
            column: Column name to analyze
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier information
        """
        col_data = self.data[column].dropna()
        
        if method == 'iqr':
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_data = self.data.loc[col_data[outliers].index].copy()
        outlier_data['outlier_type'] = method
        
        self.analysis_results['outliers'] = outlier_data
        return outlier_data


def plot_time_series(data: pd.DataFrame, time_column: str, 
                    value_columns: List[str], title: str = "Time Series Plot",
                    figsize: Tuple[int, int] = (12, 6)):
    """
    Plot time series data.
    
    Args:
        data: DataFrame with time series data
        time_column: Name of time column
        value_columns: List of value columns to plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for column in value_columns:
        plt.plot(data[time_column], data[column], label=column, alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_distribution(data: pd.DataFrame, column: str, 
                     title: str = "Distribution Plot",
                     figsize: Tuple[int, int] = (10, 6)):
    """
    Plot distribution of a column.
    
    Args:
        data: DataFrame with data
        column: Column name to plot
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    data[column].hist(bins=30, ax=axes[0], alpha=0.7)
    axes[0].set_title('Histogram')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    
    # Box plot
    data[column].plot(kind='box', ax=axes[1])
    axes[1].set_title('Box Plot')
    axes[1].set_ylabel(column)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame,
                            title: str = "Correlation Heatmap",
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Plot correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.show()


def generate_analysis_report(analyzer: ThermalDataAnalyzer, 
                           output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive analysis report.
    
    Args:
        analyzer: Fitted ThermalDataAnalyzer instance
        output_path: Optional path to save report
        
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("THERMAL SYSTEM ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Basic statistics
    if 'basic_statistics' in analyzer.analysis_results:
        report_lines.append("BASIC STATISTICS")
        report_lines.append("-" * 40)
        for column, stats in analyzer.analysis_results['basic_statistics'].items():
            report_lines.append(f"\n{column}:")
            for stat_name, value in stats.items():
                report_lines.append(f"  {stat_name}: {value:.4f}")
        report_lines.append("")
    
    # Time series analysis
    if 'time_series' in analyzer.analysis_results:
        report_lines.append("TIME SERIES ANALYSIS")
        report_lines.append("-" * 40)
        for stat_name, value in analyzer.analysis_results['time_series'].items():
            report_lines.append(f"  {stat_name}: {value:.4f}")
        report_lines.append("")
    
    # Outliers
    if 'outliers' in analyzer.analysis_results:
        report_lines.append("OUTLIER DETECTION")
        report_lines.append("-" * 40)
        num_outliers = len(analyzer.analysis_results['outliers'])
        report_lines.append(f"  Number of outliers: {num_outliers}")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def export_results(results: Dict, output_path: str, format: str = 'json'):
    """
    Export analysis results to file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Export format ('json', 'csv')
    """
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'csv':
        if isinstance(results, pd.DataFrame):
            results.to_csv(output_path, index=False)
        else:
            raise ValueError("CSV format requires DataFrame input")
    else:
        raise ValueError(f"Unknown format: {format}")
