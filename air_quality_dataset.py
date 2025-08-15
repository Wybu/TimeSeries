import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_air_quality_dataset():
    """
    Tạo dataset giả cho M5Stack AirQ theo yêu cầu cụ thể:
    - PM2.5, PM10 (bụi mịn)
    - CO₂ (ppm)
    - VOC (mg/m³)
    - Nhiệt độ (°C)
    - Độ ẩm (%RH)
    - Chuỗi thời gian liên tục, mỗi 5 phút một mẫu
    - Khoảng thời gian: 7 ngày (~2016 mẫu)
    - Dữ liệu được tạo dựa trên sóng sin + nhiễu Gaussian
    """
    print("Đang tạo dataset giả cho M5Stack AirQ...")
    
    # Tính số mẫu: 7 ngày * 24 giờ * 12 mẫu/giờ = 2016 mẫu
    n_samples = 7 * 24 * 12  # 2016 mẫu
    
    # Bắt đầu từ 00:00:00
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_samples)]
    
    data = []
    
    for i, timestamp in enumerate(timestamps):
        # Tạo pattern theo thời gian (ngày/đêm, giao thông giờ cao điểm)
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        
        # Pattern giao thông: cao điểm vào sáng (7-9h) và chiều (17-19h)
        traffic_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic_factor = 2.5
        elif 22 <= hour or hour <= 5:
            traffic_factor = 0.8
            
        # Pattern theo ngày trong tuần (cuối tuần ít giao thông hơn)
        if day_of_week >= 5:  # Thứ 7, Chủ nhật
            traffic_factor *= 0.7
        
        # Thời gian tính bằng giờ từ đầu dataset
        time_hours = i / 12.0  # 12 mẫu/giờ
        
        # 1. PM2.5 (μg/m³) - bụi mịn
        # Sóng sin cơ bản + pattern giao thông + nhiễu Gaussian
        pm25_base = 15 + 10 * np.sin(2 * np.pi * time_hours / 24)  # Pattern ngày/đêm
        pm25_traffic = traffic_factor * 15
        pm25_noise = np.random.normal(0, 2)  # Nhiễu Gaussian
        pm25 = max(5, min(150, pm25_base + pm25_traffic + pm25_noise))
        
        # 2. PM10 (μg/m³) - bụi thô
        # PM10 thường cao hơn PM2.5
        pm10_ratio = random.uniform(1.8, 2.5)
        pm10 = max(10, min(200, pm25 * pm10_ratio + np.random.normal(0, 3)))
        
        # 3. CO₂ (ppm) - carbon dioxide
        # CO2 có pattern ngày/đêm và ảnh hưởng bởi giao thông
        co2_base = 400 + 50 * np.sin(2 * np.pi * time_hours / 24)  # Pattern ngày/đêm
        co2_traffic = traffic_factor * 30
        co2_noise = np.random.normal(0, 10)
        co2 = max(350, min(800, co2_base + co2_traffic + co2_noise))
        
        # 4. VOC (mg/m³) - volatile organic compounds
        # VOC cao vào giờ cao điểm giao thông
        voc_base = 0.1 + 0.05 * np.sin(2 * np.pi * time_hours / 24)
        voc_traffic = traffic_factor * 0.2
        voc_noise = np.random.normal(0, 0.05)
        voc = max(0.01, min(1.0, voc_base + voc_traffic + voc_noise))
        
        # 5. Nhiệt độ (°C) - pattern ngày/đêm rõ ràng
        temp_base = 25 + 8 * np.sin(2 * np.pi * time_hours / 24)  # Pattern ngày/đêm
        temp_noise = np.random.normal(0, 1.5)
        temp = temp_base + temp_noise
        
        # 6. Độ ẩm (%RH) - ngược với nhiệt độ
        humidity_base = 60 - 20 * np.sin(2 * np.pi * time_hours / 24)  # Ngược với nhiệt độ
        humidity_noise = np.random.normal(0, 3)
        humidity = max(30, min(90, humidity_base + humidity_noise))
        
        # Tính AQI (Air Quality Index) đơn giản dựa trên PM2.5
        aqi = calculate_aqi(pm25)
        
        # Phân loại chất lượng không khí
        air_quality = classify_air_quality(aqi)
        
        data.append({
            'timestamp': timestamp,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'co2': round(co2, 1),
            'voc': round(voc, 4),
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'aqi': round(aqi, 1),
            'air_quality': air_quality,
            'traffic_factor': round(traffic_factor, 2)
        })
    
    df = pd.DataFrame(data)
    
    # Lưu dataset
    df.to_csv('air_quality_dataset.csv', index=False)
    
    print("Dataset đã được tạo thành công!")
    print(f"Số lượng mẫu: {len(df)}")
    print(f"Thời gian: từ {df['timestamp'].min()} đến {df['timestamp'].max()}")
    print(f"Tần suất lấy mẫu: mỗi 5 phút")
    print(f"Tổng thời gian: 7 ngày")
    
    print("\nThống kê cơ bản:")
    print(df[['pm25', 'pm10', 'co2', 'voc', 'temperature', 'humidity', 'aqi']].describe())
    
    print("\nPhân bố chất lượng không khí:")
    print(df['air_quality'].value_counts())
    
    print("\nThống kê theo giờ trong ngày:")
    hourly_stats = df.groupby(df['timestamp'].dt.hour)['pm25'].agg(['mean', 'std']).round(2)
    print(hourly_stats.head(10))
    
    return df

def calculate_aqi(pm25):
    """
    Tính AQI đơn giản dựa trên PM2.5
    """
    if pm25 <= 12.0:
        return 50 * (pm25 / 12.0)
    elif pm25 <= 35.4:
        return 50 + 50 * ((pm25 - 12.1) / (35.4 - 12.1))
    elif pm25 <= 55.4:
        return 100 + 50 * ((pm25 - 35.5) / (55.4 - 35.5))
    elif pm25 <= 150.4:
        return 150 + 100 * ((pm25 - 55.5) / (150.4 - 55.5))
    elif pm25 <= 250.4:
        return 200 + 100 * ((pm25 - 150.5) / (250.4 - 150.5))
    else:
        return 300 + 100 * ((pm25 - 250.5) / (500.4 - 250.5))

def classify_air_quality(aqi):
    """
    Phân loại chất lượng không khí dựa trên AQI
    """
    if aqi <= 50:
        return 'Tốt'
    elif aqi <= 100:
        return 'Trung bình'
    elif aqi <= 150:
        return 'Kém'
    elif aqi <= 200:
        return 'Xấu'
    elif aqi <= 300:
        return 'Rất xấu'
    else:
        return 'Nguy hiểm'

def analyze_dataset_patterns(df):
    """
    Phân tích các pattern trong dataset
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH PATTERN TRONG DATASET")
    print("="*60)
    
    # 1. Pattern theo giờ trong ngày
    print("\n1. Pattern PM2.5 theo giờ trong ngày:")
    hourly_pm25 = df.groupby(df['timestamp'].dt.hour)['pm25'].mean()
    print(hourly_pm25.round(2))
    
    # 2. Pattern theo ngày trong tuần
    print("\n2. Pattern PM2.5 theo ngày trong tuần:")
    daily_pm25 = df.groupby(df['timestamp'].dt.day_name())['pm25'].mean()
    print(daily_pm25.round(2))
    
    # 3. Tương quan giữa các thông số
    print("\n3. Ma trận tương quan:")
    correlation = df[['pm25', 'pm10', 'co2', 'voc', 'temperature', 'humidity']].corr()
    print(correlation.round(3))
    
    # 4. Thống kê theo traffic factor
    print("\n4. PM2.5 theo mức độ giao thông:")
    traffic_stats = df.groupby('traffic_factor')['pm25'].agg(['count', 'mean', 'std']).round(2)
    print(traffic_stats)
    
    return hourly_pm25, daily_pm25, correlation

if __name__ == "__main__":
    # Tạo dataset
    df = create_air_quality_dataset()
    
    # Phân tích pattern
    hourly_pm25, daily_pm25, correlation = analyze_dataset_patterns(df)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH TẠO DATASET!")
    print("="*60)
    print("File đã được lưu: air_quality_dataset.csv")
    print("Dataset sẵn sàng cho việc huấn luyện mô hình AI!")
