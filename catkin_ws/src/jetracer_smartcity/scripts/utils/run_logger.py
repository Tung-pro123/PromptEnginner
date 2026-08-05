import os
import csv
import time
from datetime import datetime

class RunLogger:
    def __init__(self, log_dir="d:/AI_Project/racing_promax/logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except Exception as e:
                print(f"Error creating log directory {self.log_dir}: {e}")
                self.log_dir = "/tmp"  # Fallback to temp if workspace is read-only
                
        # Generate log filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"run_log_{timestamp_str}.csv")
        
        # Write CSV Header
        self.headers = [
            "timestamp", 
            "fps", 
            "detected_object/sign", 
            "confidence", 
            "decision", 
            "latency_ms", 
            "control_output", 
            "event"
        ]
        
        try:
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"Logger initialized. Saving to: {self.log_file}")
        except Exception as e:
            print(f"Failed to initialize logger file: {e}")

    def log_event(self, fps, detected_object, confidence, decision, latency_ms, control_output, event=""):
        """
        Logs a single execution step to the CSV log file.
        """
        timestamp = datetime.now().isoformat()
        row = [
            timestamp,
            f"{fps:.2f}",
            str(detected_object),
            f"{confidence:.4f}",
            str(decision),
            f"{latency_ms:.2f}",
            str(control_output),
            str(event)
        ]
        
        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Error writing log entry: {e}")
