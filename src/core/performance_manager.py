"""
Performance Manager Module for Fraudulent Seller Detection Portal

This module focuses on optimizing the application's performance and scalability,
including:
- Memory management strategies (streaming, caching)
- Processing optimization (parallel processing, asynchronous tasks)
- Load balancing and performance profiling

Author: Manus AI
Version: 1.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Callable
import concurrent.futures
import threading
import time
import psutil


class PerformanceManager:
    """
    Manages and optimizes the performance and scalability of the application.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Performance Manager.
        
        Args:
            config: Configuration dictionary for performance settings.
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.max_workers = self.config.get("max_workers", os.cpu_count() or 4) # Max CPU cores or 4
        self.cache: Dict[str, Any] = {}
        self.cache_max_size = self.config.get("cache_max_size", 100) # Max items in cache
        self.cache_lru_queue: List[str] = [] # For LRU eviction
        
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the Performance Manager.
        """
        logger = logging.getLogger("performance_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/performance_manager.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def stream_data_processing(self, data_source: Any, process_chunk_func: Callable[[pd.DataFrame], Any], chunk_size: int = 10000) -> List[Any]:
        """
        Processes data in a streaming fashion to reduce memory footprint.
        
        Args:
            data_source: An iterable data source (e.g., file path, database cursor).
            process_chunk_func: A function that processes a single chunk (DataFrame).
            chunk_size: Number of rows per chunk.
            
        Returns:
            A list of results from processing each chunk.
        """
        self.logger.info(f"Starting streaming data processing with chunk size: {chunk_size}")
        results = []
        total_chunks = 0
        
        # This is a simplified example. In a real scenario, data_source would be more complex.
        # Assuming data_source is a path to a CSV for this example.
        try:
            for i, chunk in enumerate(pd.read_csv(data_source, chunksize=chunk_size)):
                self.logger.info(f"Processing chunk {i+1} (rows {i*chunk_size} to {(i+1)*chunk_size-1})")
                processed_chunk = process_chunk_func(chunk)
                results.append(processed_chunk)
                total_chunks += 1
            self.logger.info(f"Finished streaming data processing. Total chunks processed: {total_chunks}")
        except Exception as e:
            self.logger.error(f"Error during streaming data processing: {e}")
            raise PerformanceError(f"Streaming data processing failed: {str(e)}")
            
        return results
    
    def intelligent_caching(self, key: str, data: Any):
        """
        Implements an LRU (Least Recently Used) caching mechanism.
        
        Args:
            key: The key to store the data under.
            data: The data to be cached.
        """
        if key in self.cache:
            self.cache_lru_queue.remove(key)
        elif len(self.cache_lru_queue) >= self.cache_max_size:
            oldest_key = self.cache_lru_queue.pop(0)
            del self.cache[oldest_key]
            self.logger.info(f"Evicted \'{oldest_key}\' from cache (LRU).")
            
        self.cache[key] = data
        self.cache_lru_queue.append(key)
        self.logger.info(f"Cached data with key: \'{key}\\'")
        
    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieves data from the cache.
        
        Args:
            key: The key of the data to retrieve.
            
        Returns:
            The cached data, or None if not found.
        """
        if key in self.cache:
            self.cache_lru_queue.remove(key)
            self.cache_lru_queue.append(key)
            self.logger.info(f"Retrieved \'{key}\' from cache.")
            return self.cache[key]
        return None
    
    def clear_cache(self):
        """
        Clears all data from the cache.
        """
        self.cache.clear()
        self.cache_lru_queue.clear()
        self.logger.info("Cache cleared.")
        
    def parallel_process_tasks(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """
        Executes a list of tasks in parallel using a ThreadPoolExecutor.
        
        Args:
            tasks: A list of callable functions (tasks) to execute.
            
        Returns:
            A list of results from the executed tasks.
        """
        self.logger.info(f"Starting parallel processing of {len(tasks)} tasks with {self.max_workers} workers.")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Task {task.__name__ if hasattr(task, '__name__') else 'anonymous'} completed.")
                except Exception as exc:
                    self.logger.error(f"Task {task.__name__ if hasattr(task, '__name__') else 'anonymous'} generated an exception: {exc}")
                    results.append(exc) # Append exception for error handling
        self.logger.info("Parallel processing finished.")
        return results
    
    def get_system_resource_usage(self) -> Dict[str, Any]:
        """
        Retrieves current system resource usage (CPU, memory).
        
        Returns:
            A dictionary with CPU and memory usage statistics.
        """
        cpu_percent = psutil.cpu_percent(interval=1) # Blocking, waits for 1 second
        memory_info = psutil.virtual_memory()
        
        usage = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "memory_total_gb": round(memory_info.total / (1024**3), 2),
            "memory_used_gb": round(memory_info.used / (1024**3), 2)
        }
        self.logger.info(f"Current resource usage: CPU={usage['cpu_percent']}%, Memory={usage['memory_percent']}% ({usage['memory_used_gb']}GB/{usage['memory_total_gb']}GB)")
        return usage
    
    def monitor_performance(self, interval_seconds: int = 5, duration_seconds: int = 30):
        """
        Monitors system performance over a specified duration.
        
        Args:
            interval_seconds: How often to log performance metrics.
            duration_seconds: Total duration to monitor.
        """
        self.logger.info(f"Starting performance monitoring for {duration_seconds} seconds at {interval_seconds} second intervals.")
        start_time = time.time()
        metrics_history = []
        
        while (time.time() - start_time) < duration_seconds:
            metrics = self.get_system_resource_usage()
            metrics_history.append(metrics)
            time.sleep(interval_seconds)
            
        self.logger.info("Performance monitoring complete.")
        return metrics_history


class PerformanceError(Exception):
    """Custom exception for performance-related errors."""
    pass


# Example Usage (for testing purposes)
if __name__ == "__main__":
    performance_manager = PerformanceManager()
    
    # --- Test Streaming Data Processing ---
    # Create a dummy CSV file for streaming test
    dummy_streaming_data = pd.DataFrame({
        "col1": range(100000),
        "col2": np.random.rand(100000)
    })
    dummy_streaming_data.to_csv("large_data.csv", index=False)
    
    def process_chunk(chunk_df: pd.DataFrame) -> int:
        # Simulate some processing on the chunk
        return len(chunk_df)
        
    print("\n--- Testing Streaming Data Processing ---")
    try:
        chunk_lengths = performance_manager.stream_data_processing("large_data.csv", process_chunk, chunk_size=10000)
        print(f"Processed {len(chunk_lengths)} chunks. Total rows: {sum(chunk_lengths)}")
    except PerformanceError as e:
        print(f"Streaming data processing failed: {e}")
    finally:
        os.remove("large_data.csv")
        
    # --- Test Intelligent Caching ---
    print("\n--- Testing Intelligent Caching ---")
    performance_manager.intelligent_caching("key1", {"data": "value1"})
    performance_manager.intelligent_caching("key2", {"data": "value2"})
    print(f"Cached data for key1: {performance_manager.get_cached_data('key1')}")
    print(f"Cached data for key3 (not found): {performance_manager.get_cached_data('key3')}")
    
    # Fill cache to trigger LRU eviction
    for i in range(performance_manager.cache_max_size + 5):
        performance_manager.intelligent_caching(f"key_lru_{i}", {"data": i})
    print(f"Cache size after overflow: {len(performance_manager.cache)}")
    
    performance_manager.clear_cache()
    print(f"Cache size after clearing: {len(performance_manager.cache)}")
    
    # --- Test Parallel Processing ---
    print("\n--- Testing Parallel Processing ---")
    def task1():
        time.sleep(1)
        return "Task 1 Done"
        
    def task2():
        time.sleep(0.5)
        return "Task 2 Done"
        
    def task_with_error():
        raise ValueError("Simulated error in task")
        
    tasks_to_run = [task1, task2, task_with_error]
    parallel_results = performance_manager.parallel_process_tasks(tasks_to_run)
    print("Parallel processing results:", parallel_results)
    
    # --- Test System Resource Usage ---
    print("\n--- Testing System Resource Usage ---")
    resource_usage = performance_manager.get_system_resource_usage()
    print("Current System Usage:", resource_usage)
    
    # --- Test Performance Monitoring ---
    print("\n--- Testing Performance Monitoring (3 seconds) ---")
    monitor_history = performance_manager.monitor_performance(interval_seconds=1, duration_seconds=3)
    print("Performance Monitoring History:", monitor_history)
    
    # Clean up logs directory if created
    if os.path.exists("logs/performance_manager.log"):
        os.remove("logs/performance_manager.log")
    if os.path.exists("logs"):
        os.rmdir("logs")




