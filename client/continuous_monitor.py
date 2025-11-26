#!/usr/bin/env python3
"""
Continuous Monitoring Client
Monitors the prediction API and collects metrics
"""
import requests
import time
import json
from datetime import datetime
from typing import Dict, List
import statistics

class APIMonitor:
    """Monitor API performance and health"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.metrics = {
            "requests": [],
            "latencies": [],
            "errors": []
        }
    
    def check_health(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def make_prediction(self) -> Dict:
        """Make a test prediction and measure latency"""
        payload = {
            "temp": 20.0,
            "humidity": 60.0,
            "pressure": 1013.0,
            "wind_speed": 5.0
        }
        
        start = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            latency = (time.time() - start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                result = response.json()
                self.metrics["latencies"].append(latency)
                self.metrics["requests"].append({
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "latency_ms": latency,
                    "prediction": result.get("predicted_temp")
                })
                return {"status": "success", "latency_ms": latency, **result}
            else:
                self.metrics["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "status_code": response.status_code,
                    "error": response.text
                })
                return {"status": "error", "status_code": response.status_code}
        except Exception as e:
            latency = (time.time() - start) * 1000
            self.metrics["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            return {"status": "error", "error": str(e), "latency_ms": latency}
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.metrics["latencies"]:
            return {"error": "No data collected yet"}
        
        return {
            "total_requests": len(self.metrics["requests"]),
            "total_errors": len(self.metrics["errors"]),
            "success_rate": len(self.metrics["requests"]) / (len(self.metrics["requests"]) + len(self.metrics["errors"])) * 100 if (len(self.metrics["requests"]) + len(self.metrics["errors"])) > 0 else 0,
            "avg_latency_ms": statistics.mean(self.metrics["latencies"]),
            "min_latency_ms": min(self.metrics["latencies"]),
            "max_latency_ms": max(self.metrics["latencies"]),
            "median_latency_ms": statistics.median(self.metrics["latencies"]),
            "p95_latency_ms": sorted(self.metrics["latencies"])[int(len(self.metrics["latencies"]) * 0.95)] if len(self.metrics["latencies"]) > 0 else 0
        }
    
    def monitor(self, interval: int = 5, duration: int = 60):
        """Monitor API for specified duration"""
        print(f"🔍 Starting monitoring for {duration} seconds (checking every {interval}s)")
        print("=" * 60)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}")
            
            # Health check
            health = self.check_health()
            print(f"  Health: {health.get('status', 'unknown')}")
            print(f"  Model loaded: {health.get('model_loaded', False)}")
            
            # Make prediction
            result = self.make_prediction()
            if result.get("status") == "success":
                print(f"  ✅ Prediction: {result.get('predicted_temp', 'N/A')}°C")
                print(f"  ⏱️  Latency: {result.get('latency_ms', 0):.2f}ms")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
            
            # Show stats
            if iteration % 5 == 0:
                stats = self.get_stats()
                print(f"\n  📊 Stats:")
                print(f"     Requests: {stats.get('total_requests', 0)}")
                print(f"     Errors: {stats.get('total_errors', 0)}")
                print(f"     Success rate: {stats.get('success_rate', 0):.1f}%")
                print(f"     Avg latency: {stats.get('avg_latency_ms', 0):.2f}ms")
                print(f"     P95 latency: {stats.get('p95_latency_ms', 0):.2f}ms")
            
            time.sleep(interval)
        
        # Final stats
        print("\n" + "=" * 60)
        print("📊 Final Statistics:")
        stats = self.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Weather Prediction API")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    monitor = APIMonitor(base_url=args.url)
    monitor.monitor(interval=args.interval, duration=args.duration)

