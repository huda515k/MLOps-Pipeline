# Real-Time Prediction Client System

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Run Prediction Client

```bash
python client/predict_client.py
```

This will:
- Check API health
- Make multiple test predictions
- Show inference times
- Display results

### 3. Run Continuous Monitoring

```bash
# Monitor for 60 seconds, checking every 5 seconds
python client/continuous_monitor.py --duration 60 --interval 5
```

This will:
- Continuously monitor API performance
- Collect latency metrics
- Track success/error rates
- Show real-time statistics

## 📊 Usage Examples

### Basic Prediction

```python
from client.predict_client import WeatherPredictionClient

client = WeatherPredictionClient()

# Make a prediction
result = client.predict(
    temp=20.5,
    humidity=65.0,
    pressure=1013.25,
    wind_speed=5.2
)

print(f"Predicted temperature: {result['predicted_temp']}°C")
print(f"Inference time: {result['inference_time_ms']}ms")
```

### Batch Predictions

```python
from client.predict_client import WeatherPredictionClient

client = WeatherPredictionClient()

weather_data = [
    {"temp": 20.0, "humidity": 60.0, "pressure": 1013.0, "wind_speed": 5.0},
    {"temp": 25.0, "humidity": 70.0, "pressure": 1015.0, "wind_speed": 8.0},
    {"temp": 15.0, "humidity": 50.0, "pressure": 1020.0, "wind_speed": 3.0},
]

for data in weather_data:
    result = client.predict(**data)
    print(f"Input: {data['temp']}°C → Prediction: {result.get('predicted_temp', 'N/A')}°C")
```

### Performance Monitoring

```python
from client.continuous_monitor import APIMonitor

monitor = APIMonitor()

# Monitor for 2 minutes
monitor.monitor(interval=5, duration=120)

# Get statistics
stats = monitor.get_stats()
print(f"Average latency: {stats['avg_latency_ms']}ms")
print(f"Success rate: {stats['success_rate']}%")
```

## 🔧 Configuration

You can change the API URL:

```python
client = WeatherPredictionClient(base_url="http://your-api-url:8000")
```

Or via command line:

```bash
python client/continuous_monitor.py --url http://your-api-url:8000
```

## 📈 Integration Examples

### Integration with Data Pipeline

```python
import pandas as pd
from client.predict_client import WeatherPredictionClient

# Load weather data
df = pd.read_parquet("data/processed/weather_data.parquet")

# Initialize client
client = WeatherPredictionClient()

# Make predictions for each row
predictions = []
for _, row in df.iterrows():
    result = client.predict(
        temp=row['temp'],
        humidity=row['humidity'],
        pressure=row['pressure'],
        wind_speed=row['wind_speed']
    )
    predictions.append(result.get('predicted_temp', None))

# Add predictions to dataframe
df['predicted_temp'] = predictions
```

### Real-Time Monitoring Dashboard

```python
from client.continuous_monitor import APIMonitor
import time

monitor = APIMonitor()

while True:
    result = monitor.make_prediction()
    stats = monitor.get_stats()
    
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"Latency: {stats.get('avg_latency_ms', 0):.2f}ms, "
          f"Success: {stats.get('success_rate', 0):.1f}%")
    
    time.sleep(5)
```

## 🎯 Production Use

For production, you can:

1. **Add authentication**:
```python
client = WeatherPredictionClient()
client.session.headers.update({"Authorization": "Bearer YOUR_TOKEN"})
```

2. **Add retry logic**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def predict_with_retry(client, **kwargs):
    return client.predict(**kwargs)
```

3. **Add logging**:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

result = client.predict(...)
logger.info(f"Prediction: {result['predicted_temp']}°C")
```

## 📝 Notes

- The client handles timeouts (30 seconds default)
- All requests include latency measurement
- Error handling is built-in
- Results include inference time metrics

