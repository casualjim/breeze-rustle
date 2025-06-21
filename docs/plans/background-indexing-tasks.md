# Background Task Execution for Indexer

## Overview

This document outlines the plan to convert the indexer from direct execution to a background task system that ensures only one indexing operation runs at a time, using LanceDB for task tracking and persistence.

## Problem Statement

Currently, the indexer executes indexing tasks directly when `index_project` or `index_file` is called. This has several limitations:

- API requests block until indexing completes
- No way to track indexing progress or history
- Multiple simultaneous indexing operations could potentially conflict
- No recovery mechanism if the server crashes during indexing

## Solution Design

### 1. IndexTask Data Model

Create a new `IndexTask` struct and LanceDB table to track indexing tasks:

```rust
pub struct IndexTask {
    pub id: String,                    // UUID v7
    pub path: String,                  // Path to index (project or directory)
    pub status: TaskStatus,            // enum: Pending, Running, Completed, Failed
    pub created_at: NaiveDateTime,
    pub started_at: Option<NaiveDateTime>,
    pub completed_at: Option<NaiveDateTime>,
    pub error: Option<String>,         // Error message if failed
    pub files_indexed: Option<usize>,  // Number of files indexed (when completed)
}

pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}
```

### 2. Task Manager Component

Create a `TaskManager` that handles task lifecycle:

```rust
pub struct TaskManager {
    indexer: Arc<Indexer>,
    task_table: Arc<RwLock<Table>>,
}

impl TaskManager {
    // Submit a new indexing task and return its ID
    pub async fn submit_task(&self, path: String) -> Result<String>;

    // Worker loop that processes pending tasks
    pub async fn run_worker(&self, shutdown_token: CancellationToken) -> Result<()>;

    // Get task status
    pub async fn get_task(&self, task_id: &str) -> Result<Option<IndexTask>>;

    // List recent tasks
    pub async fn list_tasks(&self, limit: usize) -> Result<Vec<IndexTask>>;
}
```

### 3. Implementation Details

#### Task Submission Flow

1. API receives indexing request
2. TaskManager creates a new task record with status "Pending"
3. Task ID is returned immediately to the client
4. Client can poll task status using the ID

#### Worker Loop Flow

1. Query for oldest pending task
2. Update task status to "Running" with started_at timestamp
3. Execute the indexing operation
4. Update task with result or error, set status to "Completed" or "Failed"
5. Repeat

#### Ensuring Single Task Execution

- Use LanceDB's update operations to atomically claim tasks
- Only process tasks where status can be changed from "Pending" to "Running"
- If worker crashes, tasks remain in "Running" state and can be detected on restart

### 4. API Changes

#### Modified Endpoints

- `POST /api/v1/index/project` - Returns task ID instead of blocking

Response format:

```json
{
  "task_id": "01234567-89ab-cdef-0123-456789abcdef",
  "status": "pending"
}
```

#### New Endpoints

- `GET /api/v1/tasks/{id}` - Get task status and result
- `GET /api/v1/tasks?limit=20` - List recent tasks

Task status response:

```json
{
  "id": "01234567-89ab-cdef-0123-456789abcdef",
  "path": "/path/to/project",
  "status": "completed",
  "created_at": "2024-01-19T10:00:00Z",
  "started_at": "2024-01-19T10:00:01Z",
  "completed_at": "2024-01-19T10:05:00Z",
  "files_indexed": 42
}
```

### 5. Server Integration

In the server startup:

```rust
// Create task manager
let task_manager = TaskManager::new(indexer.clone(), task_table).await?;

// Spawn worker task
let worker_handle = tokio::spawn(async move {
    task_manager.run_worker(shutdown_token).await
});
```

### 6. Error Handling and Recovery

- **Task Failures**: Record error message, allow manual retry via API
- **Worker Crashes**: On startup, check for "Running" tasks older than a threshold and reset to "Pending"
- **Concurrent Submissions**: Queue properly using "Pending" status
- **Large Projects**: Consider adding progress tracking (files processed so far)

### 7. Future Enhancements

This design allows for future improvements:

- Priority queues (add priority field to IndexTask)
- Multiple workers (add worker_id field)
- Task cancellation (add cancelled status)
- Progress reporting (add progress percentage)
- Scheduled tasks (add scheduled_at field)

## Benefits

1. **Non-blocking API**: Clients get immediate response with task ID
2. **Task History**: Complete audit trail of all indexing operations
3. **Reliability**: Tasks persist across server restarts
4. **Scalability**: Easy to add more workers or distribute work
5. **Monitoring**: Can track indexing performance and failures

## Implementation Order

1. Create IndexTask model and LanceDB table
2. Implement TaskManager with basic submit/worker functionality
3. Modify server routes to use task submission
4. Add task status endpoints
5. Integrate worker into server startup
6. Add tests and error handling
7. Documentation and examples
