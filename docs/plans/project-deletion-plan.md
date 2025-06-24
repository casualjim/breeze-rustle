# Project Deletion Implementation Plan

## Problem

When a project directory gets deleted:
- File watcher only sees the directory deletion event, not individual files
- Project record gets deleted but documents/chunks remain in LanceDB
- Active tasks keep running for the deleted project
- File watcher keeps running
- New tasks can still be submitted

## Solution

Add a proper deletion system with a separate task queue that:
1. Immediately blocks new tasks when deletion is scheduled
2. Cancels running tasks
3. Stops the file watcher
4. Cleans up all data

## Implementation

### Phase 1: Add Project Status

**models/project.rs**
- Add `status` field (Active, PendingDeletion, Deleted)
- Add `deletion_requested_at` timestamp

**project_manager.rs**
- Add `mark_for_deletion()` method to atomically set status
- Check status in task submission

### Phase 2: Create Deletion Task System

**models/deletion_task.rs** (new file)
```rust
pub enum DeletionTaskType {
    ProjectDeletion,
    FilesDeletion { paths: Vec<PathBuf> },
}

pub struct DeletionTask {
    pub id: Uuid,
    pub project_id: Uuid,
    pub task_type: DeletionTaskType,
    pub status: TaskStatus,
    // ... timestamps, error fields
}
```

**deletion_task_manager.rs** (new file)
- Separate task manager for deletions
- `submit_deletion()` method
- `run_deletion_worker()` - processes deletion queue
- `execute_project_deletion()` - does the actual cleanup

### Phase 3: Update Task Submission

**task_manager.rs**
- In `submit_task()`: Check project status, reject if PendingDeletion
- Add `cancel_project_tasks()` to mark tasks as cancelled

**indexer.rs**
- Add `delete_project()` that:
  1. Marks project as PendingDeletion
  2. Stops file watcher
  3. Cancels active tasks
  4. Submits deletion task

### Phase 4: Fix File Watcher Bug

**file_watcher.rs**
- When directory deletion detected:
  - Query all indexed files under that path
  - Submit FilesDeletion task for those files
- Add test for directory deletion handling

### Phase 5: Update Server

**server.rs**
- Spawn deletion worker alongside indexing worker

**routes.rs**
- Update `delete_project` to use new deletion flow
- Return 202 Accepted with deletion task ID

## Key Points

- Deletions run in separate queue from indexing
- Project marked as PendingDeletion immediately
- No new work accepted once deletion scheduled
- File watcher stopped before deletion task runs
- All cleanup happens through task system (reliable)