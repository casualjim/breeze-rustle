#!/usr/bin/env python3
"""Analyze and visualize chunker performance metrics."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

def load_metrics(perf_dir="target/perf/parsing"):
    """Load all metrics from the performance directory."""
    perf_path = Path(perf_dir)
    if not perf_path.exists():
        print(f"Performance directory {perf_dir} not found!")
        return None, None
    
    # Load summary data
    summary_path = perf_path / "summary.csv"
    summary_df = None
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df['timestamp'] = pd.to_datetime(summary_df['timestamp_ms'], unit='ms')
    
    # Load raw data from all language files
    raw_data = []
    for csv_file in perf_path.glob("*.csv"):
        if csv_file.name != "summary.csv":
            df = pd.read_csv(csv_file)
            df['language'] = csv_file.stem
            raw_data.append(df)
    
    raw_df = pd.concat(raw_data) if raw_data else None
    if raw_df is not None:
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp_ms'], unit='ms')
    
    return raw_df, summary_df

def analyze_by_language(raw_df):
    """Analyze performance by language."""
    print("\n=== Performance by Language ===")
    
    # Separate parser and tokenizer data
    parser_df = raw_df[raw_df['operation'] == 'parser'].copy()
    tokenizer_df = raw_df[raw_df['operation'] == 'tokenizer'].copy()
    
    if not parser_df.empty:
        print("\nParser Performance:")
        parser_stats = parser_df.groupby('language')['duration_ms'].agg([
            'count', 'mean', 'std', 'min', 'max', 
            ('p50', lambda x: x.quantile(0.5)),
            ('p95', lambda x: x.quantile(0.95)),
            ('p99', lambda x: x.quantile(0.99))
        ]).round(3)
        print(parser_stats)
        
        # Identify slowest languages
        slowest = parser_stats.nlargest(5, 'mean')
        print(f"\nSlowest languages (by mean parse time):")
        for lang in slowest.index:
            print(f"  {lang}: {slowest.loc[lang, 'mean']:.3f}ms")
    
    if not tokenizer_df.empty:
        print("\n\nTokenizer Performance:")
        tokenizer_stats = tokenizer_df.groupby('language')['duration_ms'].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('p50', lambda x: x.quantile(0.5)),
            ('p95', lambda x: x.quantile(0.95))
        ]).round(3)
        print(tokenizer_stats)

def analyze_by_file_size(raw_df):
    """Analyze how file size affects performance."""
    print("\n\n=== Performance by File Size ===")
    
    parser_df = raw_df[raw_df['operation'] == 'parser'].copy()
    
    if not parser_df.empty:
        # Group by bucket and calculate stats
        bucket_stats = parser_df.groupby('bucket')['duration_ms'].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('p95', lambda x: x.quantile(0.95))
        ]).round(3)
        
        # Order buckets correctly
        bucket_order = ['0-1KB', '1-10KB', '10-100KB', '100-512KB', '512KB-1MB', '1-2MB', '2-4MB', '4MB+']
        bucket_stats = bucket_stats.reindex([b for b in bucket_order if b in bucket_stats.index])
        
        print("\nParser Performance by File Size:")
        print(bucket_stats)
        
        # Calculate throughput (bytes per ms)
        parser_df['throughput'] = parser_df['file_size'] / parser_df['duration_ms']
        throughput_by_bucket = parser_df.groupby('bucket')['throughput'].mean().round(1)
        print("\nAverage Throughput (bytes/ms):")
        for bucket in bucket_order:
            if bucket in throughput_by_bucket.index:
                print(f"  {bucket}: {throughput_by_bucket[bucket]:.1f}")

def plot_performance(raw_df, summary_df):
    """Create performance visualization plots."""
    if raw_df is None or raw_df.empty:
        print("No data to plot!")
        return
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Chunker Performance Analysis', fontsize=16)
    
    # 1. Parse time distribution by language
    parser_df = raw_df[raw_df['operation'] == 'parser']
    if not parser_df.empty:
        ax = axes[0, 0]
        parser_df.boxplot(column='duration_ms', by='language', ax=ax)
        ax.set_title('Parse Time Distribution by Language')
        ax.set_xlabel('Language')
        ax.set_ylabel('Duration (ms)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Parse time vs file size scatter plot
    if not parser_df.empty:
        ax = axes[0, 1]
        for lang in parser_df['language'].unique():
            lang_df = parser_df[parser_df['language'] == lang]
            ax.scatter(lang_df['file_size'], lang_df['duration_ms'], 
                      label=lang, alpha=0.6, s=30)
        ax.set_xlabel('File Size (bytes)')
        ax.set_ylabel('Parse Duration (ms)')
        ax.set_title('Parse Time vs File Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Performance over time (if we have enough data)
    if summary_df is not None and len(summary_df) > 1:
        ax = axes[1, 0]
        for lang in summary_df['language'].unique():
            lang_df = summary_df[summary_df['language'] == lang]
            ax.plot(lang_df['timestamp'], lang_df['avg_ms'], 
                   label=lang, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Duration (ms)')
        ax.set_title('Performance Over Time')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # 4. Tokenizer vs Parser comparison
    if 'operation' in raw_df.columns:
        ax = axes[1, 1]
        operation_stats = raw_df.groupby(['language', 'operation'])['duration_ms'].mean().unstack()
        if not operation_stats.empty:
            operation_stats.plot(kind='bar', ax=ax)
            ax.set_title('Parser vs Tokenizer Performance')
            ax.set_xlabel('Language')
            ax.set_ylabel('Average Duration (ms)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(['Parser', 'Tokenizer'])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("target/perf/parsing/performance_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also create a heatmap of performance by language and file size
    if not parser_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Create pivot table for heatmap
        pivot = parser_df.pivot_table(
            values='duration_ms', 
            index='language', 
            columns='bucket', 
            aggfunc='mean'
        )
        
        # Order columns correctly
        bucket_order = ['0-1KB', '1-10KB', '10-100KB', '100-512KB', '512KB-1MB', '1-2MB', '2-4MB', '4MB+']
        pivot = pivot.reindex(columns=[b for b in bucket_order if b in pivot.columns])
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Duration (ms)'})
        plt.title('Average Parse Time Heatmap (ms)')
        plt.xlabel('File Size Bucket')
        plt.ylabel('Language')
        
        heatmap_path = Path("target/perf/parsing/performance_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {heatmap_path}")
    
    plt.show()

def identify_bottlenecks(raw_df):
    """Identify specific performance bottlenecks."""
    print("\n\n=== Performance Bottlenecks ===")
    
    if raw_df is None or raw_df.empty:
        print("No data available!")
        return
    
    parser_df = raw_df[raw_df['operation'] == 'parser']
    
    if not parser_df.empty:
        # Find outliers (>95th percentile)
        p95 = parser_df['duration_ms'].quantile(0.95)
        outliers = parser_df[parser_df['duration_ms'] > p95]
        
        print(f"\nFiles taking longer than 95th percentile ({p95:.2f}ms):")
        print(f"Found {len(outliers)} outliers out of {len(parser_df)} total parses")
        
        # Group outliers by language and size
        outlier_summary = outliers.groupby(['language', 'bucket']).size()
        print("\nOutlier distribution:")
        for (lang, bucket), count in outlier_summary.items():
            print(f"  {lang} ({bucket}): {count} files")
        
        # Calculate parser/tokenizer ratio
        if 'operation' in raw_df.columns:
            # Match parser and tokenizer times for same files
            op_pivot = raw_df.pivot_table(
                values='duration_ms',
                index=['language', 'file_size'],
                columns='operation',
                aggfunc='mean'
            )
            
            if 'parser' in op_pivot.columns and 'tokenizer' in op_pivot.columns:
                op_pivot['ratio'] = op_pivot['parser'] / op_pivot['tokenizer']
                
                print("\n\nParser/Tokenizer Time Ratio by Language:")
                ratio_by_lang = op_pivot.groupby(level='language')['ratio'].agg(['mean', 'std', 'max']).round(1)
                print(ratio_by_lang)
                
                high_ratio = ratio_by_lang[ratio_by_lang['mean'] > 10]
                if not high_ratio.empty:
                    print("\nLanguages where parsing is >10x slower than tokenization:")
                    for lang in high_ratio.index:
                        print(f"  {lang}: {high_ratio.loc[lang, 'mean']:.1f}x")

def main():
    """Main analysis function."""
    print("Loading performance metrics...")
    
    perf_dir = "target/perf/parsing"
    if len(sys.argv) > 1:
        perf_dir = sys.argv[1]
    
    raw_df, summary_df = load_metrics(perf_dir)
    
    if raw_df is None:
        print("No performance data found!")
        print(f"Make sure to run the chunker first to generate data in {perf_dir}")
        return
    
    print(f"\nLoaded {len(raw_df)} raw measurements")
    if summary_df is not None:
        print(f"Loaded {len(summary_df)} summary records")
    
    # Run analyses
    analyze_by_language(raw_df)
    analyze_by_file_size(raw_df)
    identify_bottlenecks(raw_df)
    
    # Create visualizations
    try:
        plot_performance(raw_df, summary_df)
    except Exception as e:
        print(f"\nError creating plots: {e}")
        print("(This might happen if running in a non-graphical environment)")

if __name__ == "__main__":
    main()