import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def create_simple_dataset():
    """
    Tạo dataset đơn giản cho demo
    """
    print("Đang tạo dataset giả cho M5Stack AirQ...")
    
    # Tạo 100 mẫu dữ liệu
    n_samples = 100
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    data = []
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        
        # Pattern theo giờ (giao thông cao điểm)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic_factor = 2.0
        else:
            traffic_factor = 1.0
        
        # Thêm nhiễu ngẫu nhiên
        noise = random.uniform(0.8, 1.2)
        
        # PM2.5 (μg/m³)
        pm25 = max(5, 15 + (traffic_factor * 15) * noise + random.uniform(-3, 3))
        
        # PM10 (μg/m³)
        pm10 = max(10, pm25 * 1.8 + random.uniform(-2, 2))
        
        # CO (ppm)
        co = max(0.1, 0.5 + (traffic_factor * 0.8) * noise + random.uniform(-0.1, 0.1))
        
        # NO2 (ppb)
        no2 = max(5, 20 + (traffic_factor * 20) * noise + random.uniform(-5, 5))
        
        # Nhiệt độ (°C) - pattern ngày/đêm
        temp = 25 + 8 * np.sin(2 * np.pi * hour / 24) + random.uniform(-1, 1)
        
        # Độ ẩm (%)
        humidity = max(30, 60 - 15 * np.sin(2 * np.pi * hour / 24) + random.uniform(-3, 3))
        
        # Tính AQI đơn giản
        aqi = max(0, min(100, (pm25 - 5) / 30 * 100))
        
        # Phân loại
        if aqi <= 25:
            quality = 'Tốt'
        elif aqi <= 50:
            quality = 'Trung bình'
        elif aqi <= 75:
            quality = 'Kém'
        else:
            quality = 'Xấu'
        
        data.append({
            'timestamp': timestamp,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'co': round(co, 3),
            'no2': round(no2, 1),
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'aqi': round(aqi, 1),
            'quality': quality
        })
    
    df = pd.DataFrame(data)
    df.to_csv('air_quality_simple.csv', index=False)
    
    print(f"Dataset đã tạo: {len(df)} mẫu")
    print(f"Thời gian: {df['timestamp'].min()} đến {df['timestamp'].max()}")
    
    return df

def simple_time_series_analysis(df):
    """
    Phân tích time series đơn giản
    """
    print("\n" + "="*50)
    print("PHÂN TÍCH TIME SERIES ĐƠN GIẢN")
    print("="*50)
    
    # 1. Thống kê cơ bản
    print("\n1. Thống kê cơ bản:")
    print(df[['pm25', 'pm10', 'co', 'no2', 'aqi']].describe())
    
    # 2. Phân bố chất lượng không khí
    print("\n2. Phân bố chất lượng không khí:")
    quality_counts = df['quality'].value_counts()
    print(quality_counts)
    
    # 3. Pattern theo giờ
    print("\n3. Pattern theo giờ trong ngày:")
    hourly_avg = df.groupby(df['timestamp'].dt.hour)['aqi'].mean()
    print(hourly_avg)
    
    # 4. Tương quan giữa các thông số
    print("\n4. Ma trận tương quan:")
    correlation = df[['pm25', 'pm10', 'co', 'no2', 'temperature', 'humidity', 'aqi']].corr()
    print(correlation['aqi'].sort_values(ascending=False))
    
    return hourly_avg, correlation

def simple_forecasting(df):
    """
    Dự đoán đơn giản sử dụng moving average
    """
    print("\n" + "="*50)
    print("DỰ ĐOÁN ĐƠN GIẢN VỚI MOVING AVERAGE")
    print("="*50)
    
    # Tính moving average cho AQI
    df['aqi_ma_3'] = df['aqi'].rolling(window=3).mean()
    df['aqi_ma_6'] = df['aqi'].rolling(window=6).mean()
    df['aqi_ma_12'] = df['aqi'].rolling(window=12).mean()
    
    # Dự đoán đơn giản: sử dụng moving average cuối cùng
    last_ma_3 = df['aqi_ma_3'].iloc[-1]
    last_ma_6 = df['aqi_ma_6'].iloc[-1]
    last_ma_12 = df['aqi_ma_12'].iloc[-1]
    
    print(f"\nMoving Average 3 giờ: {last_ma_3:.2f}")
    print(f"Moving Average 6 giờ: {last_ma_6:.2f}")
    print(f"Moving Average 12 giờ: {last_ma_12:.2f}")
    
    # Dự đoán trung bình có trọng số
    weighted_forecast = (0.5 * last_ma_3 + 0.3 * last_ma_6 + 0.2 * last_ma_12)
    print(f"\nDự đoán có trọng số: {weighted_forecast:.2f}")
    
    # Dự đoán 3 giờ tiếp theo
    print("\nDự đoán 3 giờ tiếp theo:")
    for i in range(1, 4):
        # Đơn giản: dự đoán dựa trên trend
        trend = df['aqi'].diff().mean()
        future_aqi = weighted_forecast + (trend * i)
        future_aqi = max(0, min(100, future_aqi))
        
        future_time = df['timestamp'].iloc[-1] + timedelta(hours=i)
        print(f"  {future_time.strftime('%Y-%m-%d %H:%M')}: AQI = {future_aqi:.1f}")
    
    return df

def plot_results(df):
    """
    Vẽ biểu đồ kết quả
    """
    print("\nĐang vẽ biểu đồ...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. AQI theo thời gian
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['aqi'], label='AQI thực tế', alpha=0.7)
    ax1.plot(df['timestamp'], df['aqi_ma_3'], label='MA 3h', alpha=0.8)
    ax1.plot(df['timestamp'], df['aqi_ma_6'], label='MA 6h', alpha=0.8)
    ax1.set_title('AQI theo thời gian')
    ax1.set_xlabel('Thời gian')
    ax1.set_ylabel('AQI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. PM2.5 và PM10
    ax2 = axes[0, 1]
    ax2.plot(df['timestamp'], df['pm25'], label='PM2.5', alpha=0.7)
    ax2.plot(df['timestamp'], df['pm10'], label='PM10', alpha=0.7)
    ax2.set_title('PM2.5 và PM10 theo thời gian')
    ax2.set_xlabel('Thời gian')
    ax2.set_ylabel('μg/m³')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Nhiệt độ và độ ẩm
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    ax3.plot(df['timestamp'], df['temperature'], label='Nhiệt độ', color='red', alpha=0.7)
    ax3_twin.plot(df['timestamp'], df['humidity'], label='Độ ẩm', color='blue', alpha=0.7)
    ax3.set_title('Nhiệt độ và độ ẩm')
    ax3.set_xlabel('Thời gian')
    ax3.set_ylabel('Nhiệt độ (°C)', color='red')
    ax3_twin.set_ylabel('Độ ẩm (%)', color='blue')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Pattern theo giờ
    ax4 = axes[1, 1]
    hourly_avg = df.groupby(df['timestamp'].dt.hour)['aqi'].mean()
    ax4.bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
    ax4.set_title('AQI trung bình theo giờ')
    ax4.set_xlabel('Giờ trong ngày')
    ax4.set_ylabel('AQI trung bình')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('air_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Biểu đồ đã được lưu: air_quality_analysis.png")

def main():
    """
    Hàm chính
    """
    print("="*60)
    print("DEMO DỰ ĐOÁN CHẤT LƯỢNG KHÔNG KHÍ VỚI TIME SERIES")
    print("="*60)
    
    # 1. Tạo dataset
    df = create_simple_dataset()
    
    # 2. Phân tích time series
    hourly_avg, correlation = simple_time_series_analysis(df)
    
    # 3. Dự đoán đơn giản
    df = simple_forecasting(df)
    
    # 4. Vẽ biểu đồ
    plot_results(df)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH DEMO!")
    print("="*60)
    print("Các file đã tạo:")
    print("- air_quality_simple.csv: Dataset giả")
    print("- air_quality_analysis.png: Biểu đồ phân tích")
    print("\nĐể chạy model nâng cao, sử dụng: python air_quality_time_series.py")

if __name__ == "__main__":
    main()
