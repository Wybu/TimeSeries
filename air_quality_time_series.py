import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Time Series Models
from statsmodels.tsa.seasonal import seasonal_decompose

# Deep Learning Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Prophet (Facebook)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet không khả dụng. Cài đặt: pip install prophet")

class AirQualityPredictor:
    def __init__(self, data_path='air_quality_dataset.csv'):
        """
        Khởi tạo predictor cho dự đoán chất lượng không khí
        Theo yêu cầu: LSTM (ngắn hạn) và Prophet (dài hạn)
        """
        self.data_path = data_path
        self.df = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        
    def load_and_prepare_data(self):
        """
        Tải và chuẩn bị dữ liệu theo yêu cầu
        """
        print("Đang tải dữ liệu M5Stack AirQ...")
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Tạo các features thời gian
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['minute'] = self.df['timestamp'].dt.minute
        
        print(f"Dữ liệu đã được tải: {len(self.df)} mẫu")
        print(f"Thời gian: từ {self.df['timestamp'].min()} đến {self.df['timestamp'].max()}")
        print(f"Tần suất lấy mẫu: mỗi 5 phút")
        print(f"Tổng thời gian: 7 ngày")
        
        return self.df
    
    def create_sequences(self, data, target_col='pm25', n_steps=12):
        """
        Tạo sequences cho LSTM theo yêu cầu:
        - Input: 12 mẫu trước (1 giờ)
        - Output: dự đoán PM2.5 kế tiếp
        """
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i-n_steps:i, :-1])  # Tất cả features trừ target
            y.append(data[i, -1])  # Target (PM2.5)
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, target_col='pm25', n_steps=12, epochs=100):
        """
        Train LSTM model cho dự đoán ngắn hạn
        Theo yêu cầu: Input 12 mẫu trước (1 giờ), Output dự đoán PM2.5 kế tiếp
        """
        print(f"\n" + "="*60)
        print("HUẤN LUYỆN MÔ HÌNH LSTM (NGẮN HẠN)")
        print("="*60)
        print(f"Input: {n_steps} mẫu trước (1 giờ)")
        print(f"Output: dự đoán {target_col} kế tiếp")
        
        # Chuẩn bị dữ liệu
        features = ['pm25', 'pm10', 'co2', 'voc', 'temperature', 'humidity']
        data = self.df[features + [target_col]].values
        
        # Scale dữ liệu
        scaled_data = self.scaler.fit_transform(data)
        
        # Tạo sequences
        X, y = self.create_sequences(scaled_data, target_col, n_steps)
        
        print(f"Shape của X: {X.shape}")
        print(f"Shape của y: {y.shape}")
        
        # Split train/test (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Số mẫu training: {len(X_train)}")
        print(f"Số mẫu testing: {len(X_test)}")
        
        # Tạo model LSTM
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(n_steps, len(features))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train
        print("\nBắt đầu huấn luyện LSTM...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Đánh giá theo yêu cầu: RMSE và MAE
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nKẾT QUẢ ĐÁNH GIÁ LSTM MODEL:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        
        self.models['lstm'] = model
        self.predictions['lstm'] = {
            'y_test': y_test,
            'y_pred': y_pred.flatten(),
            'metrics': {'rmse': rmse, 'mae': mae}
        }
        
        return model, history
    
    def train_prophet_model(self, target_col='pm25'):
        """
        Train Prophet model cho dự đoán dài hạn
        Theo yêu cầu: dự báo xu hướng PM2.5 trong 24h hoặc vài ngày tới
        """
        if not PROPHET_AVAILABLE:
            print("Prophet không khả dụng")
            return None
            
        print(f"\n" + "="*60)
        print("HUẤN LUYỆN MÔ HÌNH PROPHET (DÀI HẠN)")
        print("="*60)
        print("Dự báo xu hướng PM2.5 trong 24h hoặc vài ngày tới")
        
        # Chuẩn bị dữ liệu cho Prophet
        prophet_data = self.df[['timestamp', target_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Split train/test (80% train, 20% test)
        split_idx = int(len(prophet_data) * 0.8)
        train_data = prophet_data[:split_idx]
        test_data = prophet_data[split_idx:]
        
        try:
            # Tạo và train Prophet model
            prophet_model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10
            )
            
            print("Đang huấn luyện Prophet model...")
            prophet_model.fit(train_data)
            
            # Dự đoán
            future = prophet_model.make_future_dataframe(periods=len(test_data), freq='5min')
            forecast = prophet_model.predict(future)
            
            # Lấy dự đoán cho test set
            test_forecast = forecast.tail(len(test_data))['yhat'].values
            test_actual = test_data['y'].values
            
            # Đánh giá theo yêu cầu: RMSE và MAE
            mse = mean_squared_error(test_actual, test_forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual, test_forecast)
            
            print(f"\nKẾT QUẢ ĐÁNH GIÁ PROPHET MODEL:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            
            self.models['prophet'] = prophet_model
            self.predictions['prophet'] = {
                'y_test': test_actual,
                'y_pred': test_forecast,
                'metrics': {'rmse': rmse, 'mae': mae}
            }
            
            return prophet_model
            
        except Exception as e:
            print(f"Lỗi khi train Prophet: {e}")
            return None
    
    def predict_future_lstm(self, n_steps=12):
        """
        Dự đoán tương lai với LSTM model (ngắn hạn)
        """
        if 'lstm' not in self.models:
            print("LSTM model chưa được train")
            return None
        
        print(f"\nDự đoán {n_steps} bước tiếp theo với LSTM model...")
        
        # Lấy dữ liệu cuối cùng để dự đoán
        features = ['pm25', 'pm10', 'co2', 'voc', 'temperature', 'humidity']
        last_data = self.df[features].tail(12).values
        
        # Scale dữ liệu
        scaled_data = self.scaler.transform(last_data)
        
        # Dự đoán từng bước
        predictions = []
        current_input = scaled_data.reshape(1, 12, len(features))
        
        for step in range(n_steps):
            pred = self.models['lstm'].predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Cập nhật input cho bước tiếp theo (rolling window)
            new_row = np.zeros((1, len(features)))
            new_row[0, 0] = pred  # PM2.5 là feature đầu tiên
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = new_row
        
        return predictions
    
    def predict_future_prophet(self, n_days=1):
        """
        Dự đoán tương lai với Prophet model (dài hạn)
        """
        if 'prophet' not in self.models:
            print("Prophet model chưa được train")
            return None
        
        print(f"\nDự đoán {n_days} ngày tiếp theo với Prophet model...")
        
        # Tạo future dataframe
        future = self.models['prophet'].make_future_dataframe(periods=n_days*24*12, freq='5min')
        forecast = self.models['prophet'].predict(future)
        
        # Lấy dự đoán tương lai
        future_forecast = forecast.tail(n_days*24*12)
        
        return future_forecast
    
    def plot_predictions(self, save_path=None):
        """
        Vẽ biểu đồ so sánh dự đoán
        """
        n_models = len(self.predictions)
        if n_models == 0:
            print("Không có model nào để vẽ")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. LSTM Model
        if 'lstm' in self.predictions:
            ax1 = axes[0]
            pred_data = self.predictions['lstm']
            y_test = pred_data['y_test']
            y_pred = pred_data['y_pred']
            
            ax1.plot(y_test, label='Thực tế', alpha=0.7, linewidth=1)
            ax1.plot(y_pred, label='Dự đoán LSTM', alpha=0.8, linewidth=1.5)
            ax1.set_title('LSTM Model - Dự đoán ngắn hạn (1 giờ trước → PM2.5 kế tiếp)')
            ax1.set_xlabel('Thời gian (mẫu)')
            ax1.set_ylabel('PM2.5 (μg/m³)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Hiển thị metrics
            metrics = pred_data['metrics']
            ax1.text(0.02, 0.98, f'RMSE: {metrics["rmse"]:.2f}\nMAE: {metrics["mae"]:.2f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Prophet Model
        if 'prophet' in self.predictions:
            ax2 = axes[1]
            pred_data = self.predictions['prophet']
            y_test = pred_data['y_test']
            y_pred = pred_data['y_pred']
            
            ax2.plot(y_test, label='Thực tế', alpha=0.7, linewidth=1)
            ax2.plot(y_pred, label='Dự đoán Prophet', alpha=0.8, linewidth=1.5)
            ax2.set_title('Prophet Model - Dự đoán dài hạn (xu hướng PM2.5)')
            ax2.set_xlabel('Thời gian (mẫu)')
            ax2.set_ylabel('PM2.5 (μg/m³)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Hiển thị metrics
            metrics = pred_data['metrics']
            ax2.text(0.02, 0.98, f'RMSE: {metrics["rmse"]:.2f}\nMAE: {metrics["mae"]:.2f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_models(self):
        """
        So sánh hiệu suất các model theo yêu cầu (RMSE và MAE)
        """
        if not self.predictions:
            print("Không có model nào để so sánh")
            return
        
        print("\n" + "="*60)
        print("SO SÁNH HIỆU SUẤT CÁC MODEL")
        print("="*60)
        
        results = []
        for model_name, pred_data in self.predictions.items():
            metrics = pred_data['metrics']
            results.append({
                'Model': model_name.upper(),
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE', ascending=True)  # RMSE thấp hơn = tốt hơn
        
        print(results_df.to_string(index=False))
        
        # Tìm model tốt nhất
        best_model = results_df.iloc[0]
        print(f"\nModel tốt nhất: {best_model['Model']} với RMSE = {best_model['RMSE']:.4f}")
        
        return results_df
    
    def analyze_data_patterns(self):
        """
        Phân tích pattern trong dữ liệu
        """
        print("\n" + "="*60)
        print("PHÂN TÍCH PATTERN TRONG DỮ LIỆU")
        print("="*60)
        
        # 1. Decomposition theo thời gian
        print("\n1. Phân tích decomposition của PM2.5:")
        pm25_series = self.df.set_index('timestamp')['pm25']
        
        try:
            decomposition = seasonal_decompose(pm25_series, period=288)  # 288 = 24h * 12 mẫu/giờ
            print("   - Trend: Xu hướng dài hạn")
            print("   - Seasonal: Pattern theo ngày")
            print("   - Residual: Nhiễu ngẫu nhiên")
        except:
            print("   Không thể thực hiện decomposition")
        
        # 2. Pattern theo giờ
        print("\n2. Pattern PM2.5 theo giờ trong ngày:")
        hourly_pm25 = self.df.groupby(self.df['timestamp'].dt.hour)['pm25'].agg(['mean', 'std']).round(2)
        print(hourly_pm25.head(10))
        
        # 3. Tương quan giữa các thông số
        print("\n3. Tương quan với PM2.5:")
        correlation = self.df[['pm25', 'pm10', 'co2', 'voc', 'temperature', 'humidity']].corr()
        pm25_corr = correlation['pm25'].sort_values(ascending=False)
        print(pm25_corr.round(3))
        
        return hourly_pm25, pm25_corr

def main():
    """
    Hàm chính để chạy toàn bộ pipeline theo yêu cầu
    """
    print("="*60)
    print("DỰ ĐOÁN CHẤT LƯỢNG KHÔNG KHÍ VỚI TIME SERIES AI")
    print("="*60)
    print("Theo yêu cầu: LSTM (ngắn hạn) + Prophet (dài hạn)")
    print("Target: PM2.5, Input: 12 mẫu trước (1 giờ)")
    
    # Khởi tạo predictor
    predictor = AirQualityPredictor()
    
    # Tải và chuẩn bị dữ liệu
    df = predictor.load_and_prepare_data()
    
    # Phân tích pattern
    hourly_pm25, pm25_corr = predictor.analyze_data_patterns()
    
    # Train các model theo yêu cầu
    print("\nBắt đầu huấn luyện các model...")
    
    # 1. LSTM - Mô hình ngắn hạn
    predictor.train_lstm_model(epochs=100)
    
    # 2. Prophet - Mô hình dài hạn
    if PROPHET_AVAILABLE:
        predictor.train_prophet_model()
    
    # So sánh hiệu suất
    results = predictor.compare_models()
    
    # Vẽ biểu đồ
    predictor.plot_predictions(save_path='air_quality_predictions.png')
    
    # Dự đoán tương lai
    print("\n" + "="*60)
    print("DỰ ĐOÁN TƯƠNG LAI")
    print("="*60)
    
    # LSTM: dự đoán 12 bước tiếp theo (1 giờ)
    lstm_predictions = predictor.predict_future_lstm(n_steps=12)
    if lstm_predictions is not None:
        print(f"\nLSTM - Dự đoán 12 bước tiếp theo (1 giờ):")
        for i, pred in enumerate(lstm_predictions):
            print(f"  Bước {i+1}: PM2.5 = {pred:.2f} μg/m³")
    
    # Prophet: dự đoán 1 ngày tiếp theo
    if PROPHET_AVAILABLE:
        prophet_forecast = predictor.predict_future_prophet(n_days=1)
        if prophet_forecast is not None:
            print(f"\nProphet - Dự đoán 1 ngày tiếp theo:")
            print(f"  Số mẫu dự đoán: {len(prophet_forecast)}")
            print(f"  PM2.5 trung bình: {prophet_forecast['yhat'].mean():.2f} μg/m³")
            print(f"  PM2.5 min: {prophet_forecast['yhat'].min():.2f} μg/m³")
            print(f"  PM2.5 max: {prophet_forecast['yhat'].max():.2f} μg/m³")
    
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
    print("Các model đã được huấn luyện và đánh giá:")
    print("- LSTM: Dự đoán ngắn hạn (1 giờ trước → PM2.5 kế tiếp)")
    print("- Prophet: Dự đoán dài hạn (xu hướng PM2.5)")
    print("\nChỉ số đánh giá: RMSE và MAE")
    print("File biểu đồ: air_quality_predictions.png")

if __name__ == "__main__":
    main()
