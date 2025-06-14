use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{info, error};
use std::io::Write;
use std::fs::OpenOptions;

/// Size bucket boundaries
const BUCKET_BOUNDARIES: [u64; 7] = [
    1024,       // 1KB
    10240,      // 10KB
    102400,     // 100KB
    524288,     // 512KB
    1048576,    // 1MB
    2097152,    // 2MB
    4194304,    // 4MB
];

const BUCKET_LABELS: [&str; 8] = ["0-1KB", "1-10KB", "10-100KB", "100-512KB", "512KB-1MB", "1-2MB", "2-4MB", "4MB+"];

/// Atomic statistics for a single bucket
#[derive(Default)]
struct AtomicBucketStats {
    count: AtomicU64,
    total_nanos: AtomicU64,
    min_nanos: AtomicU64,
    max_nanos: AtomicU64,
}

impl AtomicBucketStats {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_nanos: AtomicU64::new(0),
            min_nanos: AtomicU64::new(u64::MAX),
            max_nanos: AtomicU64::new(0),
        }
    }

    fn record(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        
        // Increment count
        self.count.fetch_add(1, Ordering::Relaxed);
        
        // Add to total time
        self.total_nanos.fetch_add(nanos, Ordering::Relaxed);
        
        // Update min (use compare_exchange loop for correctness)
        let mut current_min = self.min_nanos.load(Ordering::Relaxed);
        while nanos < current_min {
            match self.min_nanos.compare_exchange_weak(
                current_min,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        // Update max
        let mut current_max = self.max_nanos.load(Ordering::Relaxed);
        while nanos > current_max {
            match self.max_nanos.compare_exchange_weak(
                current_max,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    fn get_stats(&self) -> Option<(u64, Duration, Duration, Duration)> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        
        let total_nanos = self.total_nanos.load(Ordering::Relaxed);
        let min_nanos = self.min_nanos.load(Ordering::Relaxed);
        let max_nanos = self.max_nanos.load(Ordering::Relaxed);
        
        let avg_nanos = total_nanos / count;
        
        Some((
            count,
            Duration::from_nanos(avg_nanos),
            Duration::from_nanos(if min_nanos == u64::MAX { 0 } else { min_nanos }),
            Duration::from_nanos(max_nanos),
        ))
    }
}

/// Language-specific performance statistics
struct LanguageStats {
    buckets: [AtomicBucketStats; 8],
}

impl LanguageStats {
    fn new() -> Self {
        Self {
            buckets: [
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
                AtomicBucketStats::new(),
            ],
        }
    }
}

/// Tracks performance metrics for parsing operations
#[derive(Clone)]
pub struct PerformanceTracker {
    /// Map from language name to statistics
    languages: Arc<DashMap<String, Arc<LanguageStats>>>,
    /// Base directory for performance metrics
    perf_dir: Arc<std::path::PathBuf>,
    /// Append-only log file for all file processing events
    log_file: Arc<Mutex<std::fs::File>>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        // Create metrics directory in target/perf/parsing
        let perf_dir = std::path::Path::new("target/perf/parsing");
        if let Err(e) = std::fs::create_dir_all(&perf_dir) {
            error!("Failed to create performance directory: {}", e);
        }
        
        // Create append-only log file
        let log_path = perf_dir.join("processing_log.csv");
        let needs_header = !log_path.exists();
        
        let mut log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .expect("Failed to create processing log file");
            
        if needs_header {
            writeln!(log_file, "timestamp_ms,file_path,language,file_size,duration_ms,operation").ok();
        }
        
        let tracker = Self {
            languages: Arc::new(DashMap::new()),
            perf_dir: Arc::new(perf_dir.to_path_buf()),
            log_file: Arc::new(Mutex::new(log_file)),
        };
        
        // Start background reporting task
        tracker.start_reporting();
        tracker
    }
    
    /// Get or create the CSV file for a language
    fn get_language_file(&self, language: &str) -> std::io::Result<std::fs::File> {
        let safe_name = language.replace('/', "_").replace(' ', "_");
        let file_path = self.perf_dir.join(format!("{}.csv", safe_name));
        
        // Check if file exists to determine if we need to write headers
        let needs_header = !file_path.exists();
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;
        
        if needs_header {
            writeln!(file, "timestamp_ms,file_size,bucket,duration_ms,operation")?;
        }
        
        Ok(file)
    }
    
    /// Record a parsing operation
    /// Generic record method that includes operation type
    pub fn record(&self, language: String, file_size: u64, duration: Duration, operation: &str) {
        // Determine size bucket
        let bucket_idx = if file_size < 1024 {
            0 // 0-1KB
        } else if file_size < 10 * 1024 {
            1 // 1-10KB
        } else if file_size < 100 * 1024 {
            2 // 10-100KB
        } else if file_size < 512 * 1024 {
            3 // 100-512KB
        } else if file_size < 1024 * 1024 {
            4 // 512KB-1MB
        } else if file_size < 2 * 1024 * 1024 {
            5 // 1-2MB
        } else if file_size < 4 * 1024 * 1024 {
            6 // 2-4MB
        } else {
            7 // 4MB+
        };
        
        // Use operation-specific language key for stats tracking
        let stats_key = if operation == "parser" || operation == "tokenizer" {
            format!("{}_{}", language, operation)
        } else {
            language.clone()
        };
        
        // Get or create language stats
        let stats = self.languages
            .entry(stats_key)
            .or_insert_with(|| Arc::new(LanguageStats::new()))
            .clone();
        
        // Record in the appropriate bucket
        stats.buckets[bucket_idx].record(duration);
        
        // Write raw data to CSV file
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let bucket_label = BUCKET_LABELS[bucket_idx];
        
        if let Ok(mut file) = self.get_language_file(&language) {
            writeln!(
                file,
                "{},{},{},{:.3},{}",
                timestamp,
                file_size,
                bucket_label,
                duration_ms,
                operation
            ).ok();
            file.flush().ok();
        }
    }
    
    pub fn record_parse(&self, language: String, file_size: u64, duration: Duration) {
        // Determine operation type based on language suffix
        let operation = if language.ends_with("_tokenizer") {
            "tokenizer"
        } else {
            "parser"
        };
        
        self.record(language, file_size, duration, operation);
    }
    
    /// Record a file processing event with full details
    pub fn record_file_processing(&self, file_path: String, language: String, file_size: u64, duration: Duration, operation: &str) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        
        let duration_ms = duration.as_secs_f64() * 1000.0;
        
        // Write to the append-only log file
        if let Ok(mut log_file) = self.log_file.lock() {
            writeln!(
                log_file,
                "{},{},{},{},{:.3},{}",
                timestamp,
                file_path,
                language,
                file_size,
                duration_ms,
                operation
            ).ok();
            log_file.flush().ok();
        }
    }
    
    
    /// Start the background reporting task
    fn start_reporting(&self) {
        let languages = self.languages.clone();
        let perf_dir = self.perf_dir.clone();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(5));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            
            loop {
                ticker.tick().await;
                Self::write_summary(&languages, &perf_dir);
            }
        });
    }
    
    /// Write summary statistics to a file
    fn write_summary(languages: &Arc<DashMap<String, Arc<LanguageStats>>>, perf_dir: &std::path::Path) {
        if languages.is_empty() {
            return;
        }
        
        let summary_path = perf_dir.join("summary.csv");
        let needs_header = !summary_path.exists();
        
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&summary_path)
        {
            if needs_header {
                writeln!(file, "timestamp_ms,language,bucket,count,avg_ms,min_ms,max_ms").ok();
            }
            
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            
            // Sort languages for consistent output
            let mut sorted_languages: Vec<_> = languages.iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect();
            sorted_languages.sort_by(|a, b| a.0.cmp(&b.0));
            
            for (language, stats) in sorted_languages {
                for (idx, bucket) in stats.buckets.iter().enumerate() {
                    if let Some((count, avg, min, max)) = bucket.get_stats() {
                        writeln!(
                            file,
                            "{},{},{},{},{:.3},{:.3},{:.3}",
                            timestamp,
                            language,
                            BUCKET_LABELS[idx],
                            count,
                            avg.as_secs_f64() * 1000.0,
                            min.as_secs_f64() * 1000.0,
                            max.as_secs_f64() * 1000.0
                        ).ok();
                    }
                }
            }
        }
        
        info!("Performance metrics written to {}", perf_dir.display());
    }
}

/// Global performance tracker instance
static PERFORMANCE_TRACKER: std::sync::OnceLock<PerformanceTracker> = std::sync::OnceLock::new();

/// Get or create the global performance tracker
pub fn get_tracker() -> &'static PerformanceTracker {
    PERFORMANCE_TRACKER.get_or_init(PerformanceTracker::new)
}

/// A timer guard that records duration when dropped
pub struct ParseTimer {
    language: String,
    file_size: u64,
    start: Instant,
}

impl ParseTimer {
    pub fn new(language: String, file_size: u64) -> Self {
        Self {
            language,
            file_size,
            start: Instant::now(),
        }
    }
}

impl Drop for ParseTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        get_tracker().record_parse(self.language.clone(), self.file_size, duration);
    }
}

/// Timer for tokenization operations
pub struct TokenizerTimer {
    language: String,
    file_size: u64,
    start: Instant,
}

impl TokenizerTimer {
    pub fn new(language: String, file_size: u64) -> Self {
        Self {
            language,
            file_size,
            start: Instant::now(),
        }
    }
}

impl Drop for TokenizerTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        // Record with a special language tag for tokenizer timing
        get_tracker().record_parse(
            format!("{}_tokenizer", self.language), 
            self.file_size, 
            duration
        );
    }
}

/// Timer for full file processing with path tracking
pub struct FileProcessingTimer {
    file_path: String,
    language: String,
    file_size: u64,
    operation: String,
    start: Instant,
}

impl FileProcessingTimer {
    pub fn new(file_path: String, language: String, file_size: u64, operation: &str) -> Self {
        Self {
            file_path,
            language,
            file_size,
            operation: operation.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for FileProcessingTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        get_tracker().record_file_processing(
            self.file_path.clone(),
            self.language.clone(),
            self.file_size,
            duration,
            &self.operation
        );
    }
}