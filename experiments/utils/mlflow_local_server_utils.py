#!/usr/bin/env python3
"""MLflow server utilities with automatic cleanup of orphaned artifacts."""

import subprocess
import sqlite3
import os
import time
import threading
import shutil
import platform
from pathlib import Path
import stat


class MLflowDatabaseMonitor:
    """Monitor MLflow database for changes and cleanup orphaned artifacts."""

    def __init__(self, db_path, artifacts_dir):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self.last_known_runs = set()
        self.last_modified_time = 0
        self.is_windows = platform.system() == "Windows"
        self.update_known_runs()

    def update_known_runs(self):
        """Update the set of known run IDs from the database."""
        try:
            # Check if database file has been modified
            current_mtime = os.path.getmtime(self.db_path)
            if current_mtime <= self.last_modified_time:
                return

            self.last_modified_time = current_mtime

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT run_uuid FROM runs WHERE lifecycle_stage = 'active'")
            runs = {row[0] for row in cursor.fetchall()}
            conn.close()

            # Find orphaned artifacts
            orphaned_runs = self.last_known_runs - runs
            if orphaned_runs:
                print(
                    f"Found {len(orphaned_runs)} orphaned runs, cleaning up artifacts..."
                )
                self.cleanup_orphaned_artifacts(orphaned_runs)

            self.last_known_runs = runs
        except Exception as e:
            print(f"Error updating known runs: {e}")

    def _force_delete_directory(self, path, max_retries=3):
        """Force delete a directory with cross-platform retry logic."""
        for attempt in range(max_retries):
            try:
                if self.is_windows:
                    # Windows-specific handling for read-only files
                    def on_rm_error(func, path, exc_info):
                        # Make file writable and try again
                        os.chmod(path, stat.S_IWRITE)
                        func(path)

                    shutil.rmtree(path, onerror=on_rm_error)
                else:
                    # Unix-like systems (Linux, macOS)
                    # First try to make files writable if they're read-only
                    def make_writable(dirpath, dirnames, filenames):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                if not os.access(filepath, os.W_OK):
                                    os.chmod(filepath, 0o666)
                            except (OSError, PermissionError):
                                pass  # Ignore permission errors during traversal

                    # Walk through directory and make files writable
                    for root, dirs, files in os.walk(path):
                        make_writable(root, dirs, files)

                    # Now try to remove the directory
                    shutil.rmtree(path)

                return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(
                        f"Permission error on attempt {attempt + 1}, retrying in 1 second..."
                    )
                    time.sleep(1)
                else:
                    print(f"Failed to delete {path} after {max_retries} attempts: {e}")
                    return False
            except OSError as e:
                if attempt < max_retries - 1:
                    print(f"OS error on attempt {attempt + 1}, retrying in 1 second...")
                    time.sleep(1)
                else:
                    print(f"Failed to delete {path} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                print(f"Error deleting {path}: {e}")
                return False
        return False

    def cleanup_orphaned_artifacts(self, orphaned_run_ids):
        """Delete artifact directories for orphaned run IDs."""
        for run_id in orphaned_run_ids:
            artifact_dir = os.path.join(self.artifacts_dir, run_id)
            if os.path.exists(artifact_dir):
                try:
                    success = self._force_delete_directory(artifact_dir)
                    if success:
                        print(f"Cleaned up orphaned artifacts for run: {run_id}")
                    else:
                        print(f"Failed to clean up artifacts for run: {run_id}")
                except Exception as e:
                    print(f"Error cleaning up artifacts for {run_id}: {e}")

    def start_monitoring(self):
        """Start monitoring the database for changes."""
        print("Starting database monitor...")
        while True:
            try:
                self.update_known_runs()
                time.sleep(2)  # Check every 2 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error


def start_mlflow_server_with_cleanup(db_path, artifacts_dir, port="5000"):
    """Start MLflow server with automatic artifact cleanup.

    Args:
        db_path (str): Path to the SQLite database file
        artifacts_dir (str): Path to the artifacts directory
        port (str): Port number for the MLflow server (default: "5000")

    """
    # Create database monitor
    monitor = MLflowDatabaseMonitor(db_path, artifacts_dir)

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor.start_monitoring, daemon=True)
    monitor_thread.start()

    print("Starting MLflow server with automatic artifact cleanup...")
    print(f"Database: {db_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Platform: {platform.system()}")
    print("Monitoring for deleted runs and cleaning up orphaned artifacts...")
    print("Press Ctrl+C to stop the server")

    try:
        # Start MLflow server
        subprocess.run(
            [
                "mlflow",
                "ui",
                "--port",
                port,
                "--backend-store-uri",
                f"sqlite:///{db_path}",
                "--default-artifact-root",
                f"file:{artifacts_dir}",
            ]
        )
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")


if __name__ == "__main__":
    # Default behavior when run directly
    current_dir = Path(__file__).parent.parent.absolute()
    sqlite_db = current_dir / "mlflow.db"
    artifacts_dir = current_dir / "mlruns"

    start_mlflow_server_with_cleanup(sqlite_db, artifacts_dir)
