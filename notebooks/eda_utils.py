"""
EDA Utilities for Headway Prediction Analysis
Visualization and analysis functions for exploring headway datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print high-level dataset information"""
    print(f"Total rows: {len(df):,}")
    print(f"Total local track examples: {len(df.loc[df['track']=='A1']):,}")
    print(f"Total express track examples: {len(df.loc[df['track']=='A3']):,}")
    print(f"\nColumns: {df.columns.to_list()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst few rows:")
    return df.head()


def print_track_statistics(df_track: pd.DataFrame, track_name: str = "A1") -> None:
    """Print basic statistics for a specific track"""
    print("=" * 60)
    print(f"Track {track_name} - Basic Statistics")
    print("=" * 60)
    print(f"\nTotal examples: {len(df_track):,}")
    print(f"Date range: {df_track.arrival_time.min()} to {df_track.arrival_time.max()}")
    print(f"\nRoute Distribution:")
    print(df_track.route_id.value_counts())
    print(f"\nRoute percentages:")
    print(df_track.route_id.value_counts(normalize=True) * 100)


def print_missing_values(df_track: pd.DataFrame, track_name: str = "A1") -> pd.DataFrame:
    """Analyze and print missing values for a track"""
    print(f"Missing values in {track_name}:")
    print(df_track.isnull().sum())
    print(f"\nPercentage of rows with null headway: {(df_track['headway'].isnull().sum() / len(df_track)) * 100:.2f}%")
    print(f"\nNumeric feature statistics:")
    return df_track[['headway', 'time_of_day_seconds', 'hour_of_day', 'day_of_week']].describe()


def plot_headway_distribution(df_track: pd.DataFrame, track_name: str = "A1") -> Dict[str, float]:
    """
    Plot headway distribution with histogram and box plot
    Returns outlier statistics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(df_track.headway.dropna(), bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel('Headway (minutes)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{track_name} Track: Raw Headway Distribution')
    axes[0].axvline(df_track.headway.median(), color='red', linestyle='--', 
                    label=f"Median: {df_track.headway.median():.2f} min")
    axes[0].axvline(df_track.headway.mean(), color='green', linestyle='--', 
                    label=f"Mean: {df_track.headway.mean():.2f} min")
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df_track.headway.dropna(), vert=True)
    axes[1].set_ylabel('Headway (minutes)')
    axes[1].set_title(f'{track_name} Track: Headway Box Plot (Outlier Detection)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate outlier statistics
    q1 = df_track.headway.quantile(0.25)
    q3 = df_track.headway.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    n_outliers = (df_track.headway > outlier_threshold).sum()
    pct_outliers = (n_outliers / len(df_track)) * 100
    
    print(f"\nOutlier threshold (Q3 + 1.5 * IQR): {outlier_threshold:.2f} minutes")
    print(f"Number of outliers: {n_outliers}")
    print(f"Percentage of outliers: {pct_outliers:.2f}%")
    
    return {
        'q1': q1, 'q3': q3, 'iqr': iqr, 
        'threshold': outlier_threshold, 
        'n_outliers': n_outliers, 
        'pct_outliers': pct_outliers
    }


def plot_log_transformation(df_track: pd.DataFrame, track_name: str = "A1") -> pd.DataFrame:
    """
    Compare raw vs log-transformed headway distributions
    Returns dataframe with log_headway column added
    """
    df_clean = df_track[df_track['headway'].notna()].copy()
    df_clean['log_headway'] = np.log(df_clean['headway'] + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Raw headway
    axes[0].hist(df_clean['headway'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Headway (minutes)')
    axes[0].set_title(f'{track_name} Track: Raw Headway Distribution')
    axes[0].set_ylabel('Frequency')
    
    # Log-transformed headway
    axes[1].hist(df_clean['log_headway'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Log(Headway + 1)')
    axes[1].set_title(f'{track_name} Track: Log-Transformed Headway Distribution')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Raw headway - Skewness: {df_clean['headway'].skew():.3f}")
    print(f"Log headway - Skewness: {df_clean['log_headway'].skew():.3f}")
    
    return df_clean


def plot_temporal_patterns(df_clean: pd.DataFrame, track_name: str = "A1") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plot hourly temporal patterns: headway and frequency by hour of day
    Returns hourly statistics and peak/offpeak dataframes
    """
    hourly_stats = df_clean.groupby('hour_of_day')['headway'].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Mean and median headway by hour
    axes[0].plot(hourly_stats['hour_of_day'], hourly_stats['mean'], marker='o', label='Mean', linewidth=2)
    axes[0].plot(hourly_stats['hour_of_day'], hourly_stats['median'], marker='s', label='Median', linewidth=2)
    axes[0].fill_between(hourly_stats['hour_of_day'], 
                         hourly_stats['mean'] - hourly_stats['std'], 
                         hourly_stats['mean'] + hourly_stats['std'], 
                         alpha=0.2, label='±1 Std Dev')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Headway (minutes)')
    axes[0].set_title(f'{track_name} Track: Headway by Hour of Day')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24))
    
    # Train frequency by hour
    axes[1].bar(hourly_stats['hour_of_day'], hourly_stats['count'], color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Number of Arrivals')
    axes[1].set_title(f'{track_name} Track: Train Frequency by Hour')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(range(0, 24))
    
    plt.tight_layout()
    plt.show()
    
    # Peak vs off-peak analysis
    print("\nPeak hours (7-9 AM, 5-7 PM) vs Off-peak:")
    peak_hours = df_clean[df_clean['hour_of_day'].isin([7, 8, 9, 17, 18, 19])]
    offpeak_hours = df_clean[~df_clean['hour_of_day'].isin([7, 8, 9, 17, 18, 19])]
    print(f"Peak mean headway: {peak_hours['headway'].mean():.2f} min")
    print(f"Off-peak mean headway: {offpeak_hours['headway'].mean():.2f} min")
    
    return hourly_stats, peak_hours, offpeak_hours


def plot_heatmap(df_clean: pd.DataFrame, track_name: str = "A1") -> pd.DataFrame:
    """
    Plot heatmap of mean headway by hour of day and day of week
    Returns pivot table used for heatmap
    """
    heatmap_data = df_clean.pivot_table(
        values='headway', 
        index='hour_of_day', 
        columns='day_of_week', 
        aggfunc='mean'
    )
    
    day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    heatmap_data.columns = [day_labels[i-1] for i in heatmap_data.columns]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Mean Headway (minutes)'},
                linewidths=0.5, linecolor='gray')
    plt.title(f'{track_name} Track: Mean Headway by Hour and Day of Week', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Hour of Day', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print("Heatmap legend:")
    print("- Red = Longer headway (less frequent service)")
    print("- Green = Shorter headway (more frequent service)")
    
    return heatmap_data


def plot_day_of_week_analysis(df_clean: pd.DataFrame, track_name: str = "A1") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze and plot day of week patterns
    Returns dow_stats, weekday, and weekend dataframes
    """
    day_names = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 
                 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    
    dow_stats = df_clean.groupby('day_of_week')['headway'].agg(['mean', 'median', 'count']).reset_index()
    dow_stats['day_name'] = dow_stats['day_of_week'].map(day_names)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Headway by day
    axes[0].bar(dow_stats['day_name'], dow_stats['mean'], color='teal', alpha=0.7)
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('Mean Headway (minutes)')
    axes[0].set_title(f'{track_name} Track: Mean Headway by Day of Week')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Frequency by day
    axes[1].bar(dow_stats['day_name'], dow_stats['count'], color='coral', alpha=0.7)
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Number of Arrivals')
    axes[1].set_title(f'{track_name} Track: Train Frequency by Day of Week')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Weekday vs Weekend
    weekday = df_clean[df_clean['day_of_week'].isin([2, 3, 4, 5, 6])]
    weekend = df_clean[df_clean['day_of_week'].isin([1, 7])]
    
    print("\nWeekday vs Weekend:")
    print(f"Weekday mean headway: {weekday['headway'].mean():.2f} min")
    print(f"Weekend mean headway: {weekend['headway'].mean():.2f} min")
    print(f"Weekday total arrivals: {len(weekday):,}")
    print(f"Weekend total arrivals: {len(weekend):,}")
    
    return dow_stats, weekday, weekend


def plot_weekday_weekend_comparison(weekday: pd.DataFrame, weekend: pd.DataFrame, track_name: str = "A1") -> None:
    """Create comprehensive weekday vs weekend comparison visualizations"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    comparison_data = {
        'Period': ['Weekday', 'Weekend'],
        'Mean Headway (min)': [weekday['headway'].mean(), weekend['headway'].mean()],
        'Total Arrivals': [len(weekday), len(weekend)],
        'Median Headway (min)': [weekday['headway'].median(), weekend['headway'].median()]
    }
    
    # Mean headway comparison
    axes[0].bar(comparison_data['Period'], comparison_data['Mean Headway (min)'], 
                color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Mean Headway (minutes)')
    axes[0].set_title(f'{track_name} Track: Mean Headway Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparison_data['Mean Headway (min)']):
        axes[0].text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Total arrivals comparison
    axes[1].bar(comparison_data['Period'], comparison_data['Total Arrivals'], 
                color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Total Arrivals')
    axes[1].set_title(f'{track_name} Track: Service Frequency')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(comparison_data['Total Arrivals']):
        axes[1].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # Box plot comparison
    axes[2].boxplot([weekday['headway'].dropna(), weekend['headway'].dropna()], 
                    labels=['Weekday', 'Weekend'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[2].set_ylabel('Headway (minutes)')
    axes[2].set_title(f'{track_name} Track: Distribution Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0, 25)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSummary Statistics:")
    print(f"Weekday: {len(weekday):,} arrivals, {weekday['headway'].mean():.2f} min mean, {weekday['headway'].median():.2f} min median")
    print(f"Weekend: {len(weekend):,} arrivals, {weekend['headway'].mean():.2f} min mean, {weekend['headway'].median():.2f} min median")
    print(f"Difference: {((weekend['headway'].mean() - weekday['headway'].mean()) / weekday['headway'].mean() * 100):.1f}% longer headway on weekends")


def plot_track_comparison(df_a1_clean: pd.DataFrame, df_a3_clean: pd.DataFrame) -> Dict[str, float]:
    """
    Create comprehensive comparison between A1 and A3 tracks
    Returns dictionary of comparison statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # A3 Raw headway histogram
    axes[0, 0].hist(df_a3_clean['headway'], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel('Headway (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('A3 Track: Raw Headway Distribution')
    axes[0, 0].axvline(df_a3_clean['headway'].median(), color='red', linestyle='--', 
                       label=f"Median: {df_a3_clean['headway'].median():.2f}")
    axes[0, 0].axvline(df_a3_clean['headway'].mean(), color='green', linestyle='--', 
                       label=f"Mean: {df_a3_clean['headway'].mean():.2f}")
    axes[0, 0].legend()
    
    # A3 Box plot
    axes[0, 1].boxplot(df_a3_clean['headway'].dropna(), vert=True)
    axes[0, 1].set_ylabel('Headway (minutes)')
    axes[0, 1].set_title('A3 Track: Headway Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log-transformed A3
    axes[1, 0].hist(df_a3_clean['log_headway'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Log(Headway + 1)')
    axes[1, 0].set_title('A3 Track: Log-Transformed Headway')
    axes[1, 0].set_ylabel('Frequency')
    
    # Comparison of A1 vs A3 distributions
    axes[1, 1].hist(df_a1_clean['headway'], bins=50, alpha=0.5, label='A1 (Local)', edgecolor='black')
    axes[1, 1].hist(df_a3_clean['headway'], bins=50, alpha=0.5, label='A3 (Express)', edgecolor='black', color='red')
    axes[1, 1].set_xlabel('Headway (minutes)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('A1 vs A3 Headway Comparison')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 30)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate outlier statistics for A3
    q1 = df_a3_clean['headway'].quantile(0.25)
    q3 = df_a3_clean['headway'].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    
    print(f"\nRaw headway - Skewness: {df_a3_clean['headway'].skew():.3f}")
    print(f"Log headway - Skewness: {df_a3_clean['log_headway'].skew():.3f}")
    print(f"\nOutlier threshold (Q3 + 1.5 * IQR): {outlier_threshold:.2f} minutes")
    print(f"Number of outliers: {(df_a3_clean['headway'] > outlier_threshold).sum()}")
    print(f"Percentage of outliers: {((df_a3_clean['headway'] > outlier_threshold).sum() / len(df_a3_clean)) * 100:.2f}%")
    
    return {
        'a1_skew_raw': df_a1_clean['headway'].skew(),
        'a1_skew_log': df_a1_clean['log_headway'].skew(),
        'a3_skew_raw': df_a3_clean['headway'].skew(),
        'a3_skew_log': df_a3_clean['log_headway'].skew(),
        'a3_outlier_threshold': outlier_threshold
    }


def plot_autocorrelation(df_clean: pd.DataFrame, track_name: str = "A1", max_lags: int = 30) -> Dict[str, int]:
    """
    Plot autocorrelation function (ACF) to determine optimal lookback window
    Returns recommended lookback window based on significant correlations
    """
    # Remove any remaining nulls
    headway_values = df_clean['headway'].dropna().values
    
    # Calculate ACF
    acf_values = acf(headway_values, nlags=max_lags, fft=False)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot ACF using statsmodels
    plot_acf(headway_values, lags=max_lags, ax=axes[0], alpha=0.05)
    axes[0].set_xlabel('Lag (number of events)')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_title(f'{track_name} Track: Autocorrelation Function (ACF)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot ACF values as bar chart for clarity
    lags = range(max_lags + 1)
    axes[1].bar(lags, acf_values, alpha=0.7, color='steelblue')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].axhline(y=1.96/np.sqrt(len(headway_values)), color='red', linestyle='--', 
                    linewidth=1, label='95% Confidence Interval')
    axes[1].axhline(y=-1.96/np.sqrt(len(headway_values)), color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Lag (number of events)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title(f'{track_name} Track: ACF Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Determine optimal lookback window
    # Find where ACF first becomes insignificant (within confidence interval)
    confidence_threshold = 1.96 / np.sqrt(len(headway_values))
    
    # Find first lag where correlation drops below threshold
    significant_lags = []
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) > confidence_threshold:
            significant_lags.append(i)
    
    if significant_lags:
        # Recommended lookback is the last significant lag
        recommended_lookback = max(significant_lags)
    else:
        recommended_lookback = 1
    
    print(f"\nAutocorrelation Analysis:")
    print(f"Confidence threshold: ±{confidence_threshold:.4f}")
    print(f"Significant lags: {significant_lags[:10] if len(significant_lags) > 10 else significant_lags}")
    print(f"Recommended lookback window: {recommended_lookback} events")
    print(f"\nInterpretation:")
    print(f"- ACF shows how correlated current headway is with previous headways")
    print(f"- Lags outside the red confidence bands are statistically significant")
    print(f"- Recommended window captures all significant temporal dependencies")
    
    return {
        'recommended_lookback': recommended_lookback,
        'significant_lags': significant_lags,
        'acf_values': acf_values,
        'confidence_threshold': confidence_threshold
    }

