# MLflow Cleanup System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLflow Cleanup System                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Main Process                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   MLflow UI     │    │ Database Monitor│    │   Cleanup Thread    │   │
│  │   (Port 5000)   │    │   (Thread)      │    │   (Background)      │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Database Layer                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   mlflow.db     │    │   runs table    │    │   artifacts dir     │   │
│  │   (SQLite)      │    │   (metadata)    │    │   (file system)     │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Database Tracking Mechanism

### 1. File Modification Detection
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    File Modification Tracking                              │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Check mtime   │───▶│  Compare with   │───▶│  Update if changed  │   │
│  │   of mlflow.db  │    │  last_modified  │    │  last_modified_time │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Only query DB if file has changed                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Database Query Process
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Database Query Flow                                │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Connect to    │───▶│   Execute SQL   │───▶│   Extract run IDs   │   │
│  │   SQLite DB     │    │   query         │    │   from results      │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SQL Query: SELECT run_uuid FROM runs WHERE lifecycle_stage='active' │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Orphaned Run Detection
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Orphaned Run Detection                                 │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Previous      │    │   Current       │    │   Calculate         │   │
│  │   known_runs    │    │   active_runs   │    │   orphaned_runs     │   │
│  │   (set)         │    │   (set)         │    │   (set difference)  │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  orphaned_runs = last_known_runs - current_active_runs           │   │
│  │  Example: {run1, run2, run3} - {run1, run3} = {run2}            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cleanup Process Flow

### 4. Cross-Platform Cleanup Strategy
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Cleanup Strategy Selection                             │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Detect OS     │───▶│   Windows?      │───▶│   Unix-like?        │   │
│  │   platform      │    │   (Yes)         │    │   (Yes)             │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Use onerror   │    │   Pre-walk dir  │    │   Make files        │   │
│  │   callback with │    │   and chmod     │    │   writable first     │   │
│  │   stat.S_IWRITE │    │   0o666         │    │   then rmtree       │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Retry Logic for Robust Cleanup
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Retry Logic Flow                                  │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Attempt 1     │───▶│   Success?      │───▶│   Attempt 2         │   │
│  │   Delete dir     │    │   (No)          │    │   (Wait 1s)         │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │   Return True   │    │   Attempt 3     │    │   Return False      │   │
│  │   (Success)     │    │   (Wait 1s)     │    │   (Max retries)     │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           System Startup                                  │
│                                                                           │
│  1. Initialize MLflowDatabaseMonitor                                     │
│  2. Start monitoring thread (daemon)                                     │
│  3. Start MLflow UI server                                               │
│  4. Begin continuous monitoring loop                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Monitoring Loop (Every 2s)                         │
│                                                                           │
│  1. Check if mlflow.db file has been modified                            │
│  2. If modified, query database for active runs                          │
│  3. Compare with previously known runs                                   │
│  4. Identify orphaned runs                                               │
│  5. Clean up orphaned artifact directories                               │
│  6. Update known runs set                                                │
│  7. Sleep for 2 seconds, repeat                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components Explained

### Database Monitoring
- **File Modification Tracking**: Uses `os.path.getmtime()` to detect when the SQLite database file has been modified
- **Efficient Queries**: Only queries the database when changes are detected
- **Connection Management**: Properly opens and closes SQLite connections

### Orphaned Detection Logic
- **Set Operations**: Uses Python sets for efficient difference calculations
- **State Tracking**: Maintains `last_known_runs` to track previous state
- **Delta Detection**: Only processes runs that have been removed since last check

### Cross-Platform Cleanup
- **OS Detection**: Uses `platform.system()` to determine appropriate strategy
- **Windows Strategy**: Handles read-only files with `onerror` callback
- **Unix Strategy**: Pre-walks directories to set proper permissions
- **Retry Logic**: Handles temporary file locks and permission issues

### Threading Architecture
- **Daemon Thread**: Monitor runs as background process
- **Non-blocking**: Main MLflow server continues running normally
- **Graceful Shutdown**: Thread automatically terminates when main process ends

## Benefits of This Approach

1. **Automatic Cleanup**: No manual intervention required
2. **Cross-Platform**: Works on Windows, Linux, and macOS
3. **Efficient**: Only processes changes when database is modified
4. **Robust**: Handles permission errors and file locks
5. **Non-Intrusive**: Doesn't interfere with MLflow operations
6. **Real-time**: Continuously monitors for orphaned artifacts 