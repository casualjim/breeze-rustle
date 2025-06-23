# Embedding Failure Retry Mechanism

## Problem Statement

Currently, the breeze-indexer detects embedding failures in `document_builder_task` but only logs them without retry. When embedding services fail (rate limits, network issues, etc.), affected files are skipped and never retried. This results in incomplete indexes that require manual intervention.

## Solution Overview

Implement a separate retry queue that processes failed embeddings when the main task queue is empty. This ensures retry tasks don't starve new indexing tasks while providing automatic recovery from transient failures.

## Design

### Data Structures

#### FailedEmbeddingBatch Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryError {
    pub timestamp: chrono::NaiveDateTime,
    pub error: String,
}

pub struct FailedEmbeddingBatch {
    pub id: Uuid,
    pub task_id: Uuid,
    pub project_id: Uuid,
    pub project_path: String,
    pub failed_files: BTreeSet<String>,  // In-memory representation
    pub errors: Vec<RetryError>,         // Error history
    pub retry_count: i32,
    pub retry_after: chrono::NaiveDateTime,
    pub created_at: chrono::NaiveDateTime,
}
```

**Storage Notes:**

- `failed_files` stored as sorted ListArray in Arrow/LanceDB
- `errors` stored as JSON string in Arrow/LanceDB
- Using BTreeSet for failed_files leverages memory efficiency for paths with common prefixes

### Retry Schedule

Exponential backoff with cap:

- Initial failure → retry after 1 minute
- 1st retry fails → retry after 5 minutes
- 2nd retry fails → retry after 10 minutes
- 3rd retry fails → retry after 20 minutes
- 4th retry fails → retry after 40 minutes
- 5+ retries fail → retry every 60 minutes (cap)

```rust
fn calculate_next_retry_after(retry_count: i32) -> chrono::NaiveDateTime {
    let minutes = match retry_count {
        0 => 1,
        1 => 5,
        2 => 10,
        3 => 20,
        4 => 40,
        _ => 60,
    };
    chrono::Utc::now().naive_utc() + chrono::Duration::minutes(minutes)
}
```

### Task Processing Flow

1. **Main Queue Priority**: Always process pending tasks first
2. **Retry Queue**: When main queue is empty:
   - Query failed batches where `retry_after <= now`
   - Take oldest eligible batch
   - Create PartialUpdate task with failed files
   - Update batch with new retry_after time
3. **Failure Handling**:
   - Accumulate all failures during task execution
   - Store as single FailedEmbeddingBatch
   - Append error to history on each retry

### Implementation Steps

#### 1. Add FailedEmbeddingBatch Model (models.rs)

- Define struct with proper Arrow schema
- Implement `from_record_batch` and `into_arrow` methods
- Add table creation methods

#### 2. Update bulk_indexer.rs

- Track failures in `document_builder_task` using BTreeSet
- Return `(success_count, Option<(BTreeSet<String>, String)>)`
- Aggregate all BatchFailure events into single failure set

#### 3. Update task_manager.rs

- Add `failed_batches_table: Arc<RwLock<Table>>` field
- Modify `process_next_task()` to check retry queue when main queue empty
- Add `process_retry_task()` method:

  ```rust
  async fn process_retry_task(&self) -> Result<bool, IndexerError> {
      // Query batches ready for retry
      let now = chrono::Utc::now().naive_utc();
      let filter = format!("retry_after <= '{}'", now);

      // Get oldest eligible batch
      // Create PartialUpdate task
      // Update retry_count and retry_after
      // Return true if task created
  }
  ```

#### 4. Update Task Completion

- Add TaskStatus::PartiallyCompleted
- When task completes with failures:
  - Create FailedEmbeddingBatch record
  - Set initial retry_after based on schedule
  - Mark task as PartiallyCompleted

#### 5. Handle Retry Failures

- If retry task fails again:
  - Find existing FailedEmbeddingBatch by task_id
  - Append new RetryError to errors list
  - Increment retry_count
  - Update retry_after based on new count

## Benefits

1. **No Task Starvation**: Retry only when main queue empty
2. **Exponential Backoff**: Prevents hammering failing services
3. **Error History**: Full debugging context preserved
4. **Memory Efficient**: BTreeSet leverages path prefix sharing
5. **Persistent**: Survives service restarts
6. **Self-Healing**: Continues retrying until successful

## Testing Strategy

1. Unit tests for retry schedule calculation
2. Integration test with simulated embedding failures
3. Test retry task creation and processing
4. Verify error history accumulation
5. Test persistence across restarts

## Monitoring

Track metrics:

- Number of failed batches by project
- Retry counts distribution
- Time to successful retry
- Permanent failure detection (high retry count)
