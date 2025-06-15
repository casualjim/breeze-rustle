#!/usr/bin/env python3
"""Analyze and visualize chunker performance metrics using polars."""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


def load_metrics(perf_dir="target/perf/parsing"):
    """Load all metrics from the performance directory."""
    perf_path = Path(perf_dir)
    if not perf_path.exists():
        print(f"Performance directory {perf_dir} not found!")
        return None, None, None

    # Load summary data
    summary_path = perf_path / "summary.csv"
    summary_df = None
    if summary_path.exists():
        summary_df = pl.read_csv(summary_path)
        summary_df = summary_df.with_columns(
            [(pl.col("timestamp_ms") * 1000).cast(pl.Datetime).alias("timestamp")]
        )

    # Load raw data from all language files
    raw_dfs = []
    for csv_file in perf_path.glob("*.csv"):
        if csv_file.name not in ["summary.csv", "processing_log.csv"]:
            try:
                # Try reading with explicit schema to handle any formatting issues
                df = pl.read_csv(
                    csv_file,
                    schema_overrides={
                        "timestamp_ms": pl.Int64,
                        "file_size": pl.Int64,
                        "bucket": pl.Utf8,
                        "duration_ms": pl.Float64,
                        "operation": pl.Utf8,
                    },
                    ignore_errors=True,  # Skip malformed lines
                )
                df = df.with_columns(
                    [
                        pl.lit(csv_file.stem).alias("language"),
                        (pl.col("timestamp_ms") * 1000)
                        .cast(pl.Datetime)
                        .alias("timestamp"),
                    ]
                )
                raw_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")
                continue

    raw_df = pl.concat(raw_dfs) if raw_dfs else None

    # Load processing log if available
    log_path = perf_path / "processing_log.csv"
    log_df = None
    if log_path.exists():
        log_df = pl.read_csv(log_path)
        log_df = log_df.with_columns(
            [(pl.col("timestamp_ms") * 1000).cast(pl.Datetime).alias("timestamp")]
        )

    return raw_df, summary_df, log_df


def analyze_by_language(raw_df):
    """Analyze performance by language."""
    print("\n=== Performance by Language ===")

    if raw_df is None or raw_df.is_empty():
        print("No data available!")
        return

    # Separate parser and tokenizer data
    parser_df = raw_df.filter(pl.col("operation") == "parser")
    tokenizer_df = raw_df.filter(pl.col("operation") == "tokenizer")

    if not parser_df.is_empty():
        print("\nParser Performance:")
        parser_stats = (
            parser_df.group_by("language")
            .agg(
                [
                    pl.count("duration_ms").alias("count"),
                    pl.mean("duration_ms").alias("mean"),
                    pl.std("duration_ms").alias("std"),
                    pl.min("duration_ms").alias("min"),
                    pl.max("duration_ms").alias("max"),
                    pl.quantile("duration_ms", 0.5).alias("p50"),
                    pl.quantile("duration_ms", 0.95).alias("p95"),
                    pl.quantile("duration_ms", 0.99).alias("p99"),
                ]
            )
            .sort("mean", descending=True)
        )

        print(parser_stats)

        # Identify slowest languages
        slowest = parser_stats.head(5)
        print("\nSlowest languages (by mean parse time):")
        for row in slowest.rows():
            print(f"  {row[0]}: {row[2]:.3f}ms")

    if not tokenizer_df.is_empty():
        print("\n\nTokenizer Performance:")
        tokenizer_stats = (
            tokenizer_df.group_by("language")
            .agg(
                [
                    pl.count("duration_ms").alias("count"),
                    pl.mean("duration_ms").alias("mean"),
                    pl.std("duration_ms").alias("std"),
                    pl.min("duration_ms").alias("min"),
                    pl.max("duration_ms").alias("max"),
                    pl.quantile("duration_ms", 0.5).alias("p50"),
                    pl.quantile("duration_ms", 0.95).alias("p95"),
                ]
            )
            .sort("mean", descending=True)
        )

        print(tokenizer_stats)


def analyze_by_file_size(raw_df):
    """Analyze how file size affects performance."""
    print("\n\n=== Performance by File Size ===")

    if raw_df is None or raw_df.is_empty():
        print("No data available!")
        return

    parser_df = raw_df.filter(pl.col("operation") == "parser")

    if not parser_df.is_empty():
        # Group by bucket and calculate stats
        bucket_stats = parser_df.group_by("bucket").agg(
            [
                pl.count("duration_ms").alias("count"),
                pl.mean("duration_ms").alias("mean"),
                pl.std("duration_ms").alias("std"),
                pl.min("duration_ms").alias("min"),
                pl.max("duration_ms").alias("max"),
                pl.quantile("duration_ms", 0.95).alias("p95"),
            ]
        )

        # Order buckets correctly
        bucket_order = [
            "0-1KB",
            "1-10KB",
            "10-100KB",
            "100-512KB",
            "512KB-1MB",
            "1-2MB",
            "2-4MB",
            "4MB+",
        ]
        bucket_stats = bucket_stats.with_columns(
            [pl.col("bucket").cast(pl.Enum(bucket_order))]
        ).sort("bucket")

        print("\nParser Performance by File Size:")
        print(bucket_stats)

        # Calculate throughput (bytes per ms)
        parser_with_throughput = parser_df.with_columns(
            [(pl.col("file_size") / pl.col("duration_ms")).alias("throughput")]
        )

        throughput_by_bucket = (
            parser_with_throughput.group_by("bucket")
            .agg([pl.mean("throughput").alias("avg_throughput")])
            .with_columns([pl.col("bucket").cast(pl.Enum(bucket_order))])
            .sort("bucket")
        )

        print("\nAverage Throughput (bytes/ms):")
        for row in throughput_by_bucket.rows():
            print(f"  {row[0]}: {row[1]:.1f}")


def analyze_processing_log(log_df):
    """Analyze the file processing log."""
    print("\n\n=== File Processing Analysis ===")

    if log_df is None or log_df.is_empty():
        print("No processing log data available!")
        return

    print(f"\nTotal files processed: {len(log_df)}")

    # Group by language
    language_stats = (
        log_df.group_by("language")
        .agg(
            [
                pl.count("duration_ms").alias("count"),
                pl.mean("duration_ms").alias("mean_ms"),
                pl.sum("file_size").alias("total_bytes"),
                pl.mean("file_size").alias("avg_file_size"),
            ]
        )
        .sort("mean_ms", descending=True)
    )

    print("\nProcessing time by language:")
    print(language_stats)

    # Find slowest files
    slowest_files = log_df.sort("duration_ms", descending=True).head(10)
    print("\nSlowest files to process:")
    for row in slowest_files.rows():
        file_path = row[1]
        language = row[2]
        size = row[3]
        duration = row[4]
        print(f"  {file_path} ({language}, {size / 1024:.1f}KB): {duration:.1f}ms")


def plot_performance(raw_df, summary_df, log_df):
    """Create performance visualization plots."""
    if raw_df is None or raw_df.is_empty():
        print("No data to plot!")
        return

    # Set up the plot style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Chunker Performance Analysis", fontsize=16)

    # 1. Parse time distribution by language (box plot)
    parser_df = raw_df.filter(pl.col("operation") == "parser")
    if not parser_df.is_empty():
        ax = axes[0, 0]

        # Convert to pandas for matplotlib compatibility
        parser_pd = parser_df.to_pandas()
        parser_pd.boxplot(column="duration_ms", by="language", ax=ax)
        ax.set_title("Parse Time Distribution by Language")
        ax.set_xlabel("Language")
        ax.set_ylabel("Duration (ms)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # 2. Parse time vs file size scatter plot
    if not parser_df.is_empty():
        ax = axes[0, 1]

        # Plot each language separately
        languages = parser_df.select("language").unique().to_series().to_list()
        for lang in languages:
            lang_data = parser_df.filter(pl.col("language") == lang).to_pandas()
            ax.scatter(
                lang_data["file_size"],
                lang_data["duration_ms"],
                label=lang,
                alpha=0.6,
                s=30,
            )

        ax.set_xlabel("File Size (bytes)")
        ax.set_ylabel("Parse Duration (ms)")
        ax.set_title("Parse Time vs File Size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 3. File processing timeline (if log data available)
    if log_df is not None and not log_df.is_empty():
        ax = axes[1, 0]

        # Convert to pandas and plot timeline
        log_pd = log_df.to_pandas()
        import pandas as pd

        log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])

        # Group by language and plot cumulative files processed
        for lang in log_pd["language"].unique():
            lang_data = log_pd[log_pd["language"] == lang].sort_values("timestamp")
            lang_data["cumulative"] = range(1, len(lang_data) + 1)
            ax.plot(
                lang_data["timestamp"],
                lang_data["cumulative"],
                label=lang,
                marker="o",
                markersize=2,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Files Processed")
        ax.set_title("File Processing Timeline")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    # 4. Parser vs Tokenizer comparison
    if "operation" in raw_df.columns:
        ax = axes[1, 1]

        # Calculate average times by language and operation
        operation_stats = (
            raw_df.group_by(["language", "operation"])
            .agg([pl.mean("duration_ms").alias("avg_duration")])
            .pivot(values="avg_duration", index="language", columns="operation")
        )

        if not operation_stats.is_empty():
            operation_pd = operation_stats.to_pandas()
            operation_pd.plot(kind="bar", ax=ax)
            ax.set_title("Parser vs Tokenizer Performance")
            ax.set_xlabel("Language")
            ax.set_ylabel("Average Duration (ms)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(["Parser", "Tokenizer"])

    plt.tight_layout()

    # Save the plot
    output_path = Path("target/perf/parsing/performance_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Create performance heatmap
    if not parser_df.is_empty():
        plt.figure(figsize=(10, 6))

        # Create pivot table for heatmap
        pivot_df = (
            parser_df.group_by(["language", "bucket"])
            .agg([pl.mean("duration_ms").alias("avg_duration")])
            .pivot(values="avg_duration", index="language", columns="bucket")
        )

        # Order columns correctly
        bucket_order = [
            "0-1KB",
            "1-10KB",
            "10-100KB",
            "100-512KB",
            "512KB-1MB",
            "1-2MB",
            "2-4MB",
            "4MB+",
        ]
        available_buckets = [b for b in bucket_order if b in pivot_df.columns]
        pivot_df = pivot_df.select(["language"] + available_buckets)

        # Convert to pandas for seaborn
        pivot_pd = pivot_df.to_pandas()
        pivot_pd.set_index("language", inplace=True)

        # Create heatmap
        sns.heatmap(
            pivot_pd,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Duration (ms)"},
        )
        plt.title("Average Parse Time Heatmap (ms)")
        plt.xlabel("File Size Bucket")
        plt.ylabel("Language")

        heatmap_path = Path("target/perf/parsing/performance_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {heatmap_path}")

    plt.show()


def identify_bottlenecks(raw_df, log_df):
    """Identify specific performance bottlenecks."""
    print("\n\n=== Performance Bottlenecks ===")

    if raw_df is None or raw_df.is_empty():
        print("No data available!")
        return

    parser_df = raw_df.filter(pl.col("operation") == "parser")

    if not parser_df.is_empty():
        # Find outliers (>95th percentile)
        p95 = parser_df.select(pl.quantile("duration_ms", 0.95)).item()
        outliers = parser_df.filter(pl.col("duration_ms") > p95)

        print(f"\nFiles taking longer than 95th percentile ({p95:.2f}ms):")
        print(f"Found {len(outliers)} outliers out of {len(parser_df)} total parses")

        # Group outliers by language and size
        outlier_summary = (
            outliers.group_by(["language", "bucket"])
            .agg([pl.count("duration_ms").alias("count")])
            .sort(["language", "bucket"])
        )

        print("\nOutlier distribution:")
        for row in outlier_summary.rows():
            print(f"  {row[0]} ({row[1]}): {row[2]} files")

        # Calculate parser/tokenizer ratio
        if "operation" in raw_df.columns:
            # Join parser and tokenizer data
            parser_agg = (
                raw_df.filter(pl.col("operation") == "parser")
                .group_by(["language", "file_size"])
                .agg([pl.mean("duration_ms").alias("parser_ms")])
            )

            tokenizer_agg = (
                raw_df.filter(pl.col("operation") == "tokenizer")
                .group_by(["language", "file_size"])
                .agg([pl.mean("duration_ms").alias("tokenizer_ms")])
            )

            joined = parser_agg.join(
                tokenizer_agg, on=["language", "file_size"], how="inner"
            )
            joined = joined.with_columns(
                [(pl.col("parser_ms") / pl.col("tokenizer_ms")).alias("ratio")]
            )

            # Average ratio by language
            ratio_by_lang = (
                joined.group_by("language")
                .agg(
                    [
                        pl.mean("ratio").alias("mean_ratio"),
                        pl.std("ratio").alias("std_ratio"),
                        pl.max("ratio").alias("max_ratio"),
                    ]
                )
                .sort("mean_ratio", descending=True)
            )

            print("\n\nParser/Tokenizer Time Ratio by Language:")
            print(ratio_by_lang)

            high_ratio = ratio_by_lang.filter(pl.col("mean_ratio") > 10)
            if not high_ratio.is_empty():
                print("\nLanguages where parsing is >10x slower than tokenization:")
                for row in high_ratio.rows():
                    print(f"  {row[0]}: {row[1]:.1f}x")

    # Analyze processing log for patterns
    if log_df is not None and not log_df.is_empty():
        print("\n\nFile Processing Patterns:")

        # Find files that took unusually long relative to their size
        log_with_throughput = log_df.with_columns(
            [(pl.col("file_size") / pl.col("duration_ms")).alias("throughput")]
        )

        # Find bottom 10% throughput
        p10_throughput = log_with_throughput.select(
            pl.quantile("throughput", 0.1)
        ).item()
        slow_files = log_with_throughput.filter(pl.col("throughput") < p10_throughput)

        print(f"\nFiles with bottom 10% throughput (<{p10_throughput:.1f} bytes/ms):")
        slow_sorted = slow_files.sort("throughput").head(10)
        for row in slow_sorted.rows():
            file_path = row[1]
            language = row[2]
            size = row[3]
            throughput = row[6]
            print(
                f"  {file_path} ({language}, {size / 1024:.1f}KB): {throughput:.1f} bytes/ms"
            )


def main():
    """Main analysis function."""
    print("Loading performance metrics...")

    perf_dir = "target/perf/parsing"
    if len(sys.argv) > 1:
        perf_dir = sys.argv[1]

    raw_df, summary_df, log_df = load_metrics(perf_dir)

    if raw_df is None:
        print("No performance data found!")
        print(f"Make sure to run the chunker first to generate data in {perf_dir}")
        return

    print(f"\nLoaded {len(raw_df)} raw measurements")
    if summary_df is not None:
        print(f"Loaded {len(summary_df)} summary records")
    if log_df is not None:
        print(f"Loaded {len(log_df)} file processing records")

    # Run analyses
    analyze_by_language(raw_df)
    analyze_by_file_size(raw_df)
    analyze_processing_log(log_df)
    identify_bottlenecks(raw_df, log_df)

    # Create visualizations
    try:
        plot_performance(raw_df, summary_df, log_df)
    except ImportError:
        print("\nNote: pandas is needed for some visualizations. Skipping plots.")
    except Exception as e:
        print(f"\nError creating plots: {e}")
        print("(This might happen if running in a non-graphical environment)")


if __name__ == "__main__":
    main()
