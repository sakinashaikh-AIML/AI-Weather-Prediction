import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL TRAINING SCRIPT - FIXED VERSION
# ============================================================================

class ModelTrainer:
    """Trains weather prediction models including AQI"""
    
    def __init__(self):
        self.models = {}
    
    def generate_training_data(self, samples=5000):
        """Generate synthetic training data with AQI"""
        np.random.seed(42)
        
        print(f"🔄 Generating {samples} training samples...")
        
        # Generate features
        data = {
            'temperature': np.random.uniform(-10, 40, samples),
            'humidity': np.random.uniform(20, 100, samples),
            'pressure': np.random.uniform(970, 1050, samples),
            'wind_speed': np.random.exponential(5, samples),
            'wind_deg': np.random.uniform(0, 360, samples),
            'cloud_cover': np.random.uniform(0, 100, samples),
            'visibility': np.random.exponential(10000, samples),
            'month': np.random.randint(1, 13, samples),
            'hour': np.random.randint(0, 24, samples),
            'day_of_year': np.random.randint(1, 366, samples),
            'latitude': np.random.uniform(-90, 90, samples),
            'longitude': np.random.uniform(-180, 180, samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        df['pressure_normalized'] = (df['pressure'] - 1013) / 20
        df['wind_pressure'] = df['wind_speed'] * df['pressure_normalized']
        df['visibility_km'] = df['visibility'] / 1000
        df['is_day'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        # Generate current AQI
        urban_factor = 1 + (abs(df['latitude']) < 45).astype(int) * 0.5
        industrial_factor = 1 + (df['longitude'] > 0).astype(int) * 0.3
        temp_factor = np.clip(df['temperature'] / 25, 0.8, 1.5)
        wind_factor = np.clip(10 / (df['wind_speed'] + 1), 0.5, 2.0)
        humidity_factor = 1 + (df['humidity'] / 100) * 0.5
        pressure_factor = 1 + ((1013 - df['pressure']) / 100) * 0.2
        
        df['current_aqi'] = (
            50 * urban_factor * industrial_factor * 
            temp_factor * wind_factor * humidity_factor * pressure_factor +
            np.random.normal(0, 20, samples)
        )
        df['current_aqi'] = np.clip(df['current_aqi'], 0, 500)
        
        # Generate pollutant concentrations
        df['pm2_5'] = df['current_aqi'] / 10 + np.random.normal(0, 5, samples)
        df['pm10'] = df['current_aqi'] / 8 + np.random.normal(0, 8, samples)
        df['pm2_5'] = np.clip(df['pm2_5'], 0, 500)
        df['pm10'] = np.clip(df['pm10'], 0, 600)
        
        # Generate target variables (next day's weather)
        # Temperature target
        seasonal_factor = 10 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['next_day_temp'] = (
            0.6 * df['temperature'] + 
            0.1 * seasonal_factor + 
            0.1 * (df['month'] - 6.5) +
            np.random.normal(0, 2, samples)
        )
        
        # Humidity target - FIXED: Added missing humidity target
        df['next_day_humidity'] = (
            0.7 * df['humidity'] + 
            0.2 * np.cos(2 * np.pi * df['day_of_year'] / 365) * 20 +
            np.random.normal(0, 5, samples)
        )
        df['next_day_humidity'] = np.clip(df['next_day_humidity'], 20, 100)
        
        # Precipitation targets - FIXED: Make sure these columns exist
        rain_base = (df['humidity'] / 100) * 0.3 + (df['cloud_cover'] / 100) * 0.4
        rain_seasonal = 0.2 * np.sin(2 * np.pi * df['day_of_year'] / 365 + np.pi/2)
        df['rain_probability'] = np.clip(rain_base + rain_seasonal + np.random.normal(0, 0.1, samples), 0, 1)
        
        # Precipitation amount (regression target)
        df['next_day_precipitation'] = df['rain_probability'] * np.random.exponential(2, samples)
        
        # Precipitation classification (binary: will it rain?)
        df['will_rain'] = (df['rain_probability'] > 0.3).astype(int)
        
        # AQI target (next day)
        aqi_persistence = 0.7
        temp_change = (df['next_day_temp'] - df['temperature']) / 10
        wind_change = (np.random.exponential(5, samples) - df['wind_speed']) / 5
        
        df['next_day_aqi'] = (
            aqi_persistence * df['current_aqi'] +
            20 * temp_change +
            15 * wind_change +
            np.random.normal(0, 15, samples)
        )
        df['next_day_aqi'] = np.clip(df['next_day_aqi'], 0, 500)
        
        # AQI category target (for classification)
        aqi_categories = {
            0: {'min': 0, 'max': 50},
            1: {'min': 51, 'max': 100},
            2: {'min': 101, 'max': 150},
            3: {'min': 151, 'max': 200},
            4: {'min': 201, 'max': 300},
            5: {'min': 301, 'max': 500}
        }
        
        df['next_day_aqi_category'] = 0
        for i in range(6):
            mask = (df['next_day_aqi'] >= aqi_categories[i]['min']) & (df['next_day_aqi'] <= aqi_categories[i]['max'])
            df.loc[mask, 'next_day_aqi_category'] = i
        
        print(f"✅ Generated {samples} training samples")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def train_models(self, df):
        """Train ML models on the data - COMPLETELY FIXED VERSION"""
        print("🔄 Starting model training...")
        
        # Feature columns
        feature_cols = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg',
            'cloud_cover', 'visibility', 'month', 'hour', 'day_of_year', 'is_day',
            'latitude', 'longitude', 'temp_humidity_ratio', 'pressure_normalized',
            'wind_pressure', 'visibility_km'
        ]
        
        # Add current AQI features if available
        if 'current_aqi' in df.columns:
            # Create interaction features
            df['aqi_temp_interaction'] = df['current_aqi'] * df['temperature'] / 100
            df['aqi_humidity_interaction'] = df['current_aqi'] * df['humidity'] / 100
            feature_cols.extend(['current_aqi', 'pm2_5', 'pm10', 
                                'aqi_temp_interaction', 'aqi_humidity_interaction'])
        
        X = df[feature_cols]
        
        # Targets - FIXED: Make sure all these columns exist
        print("🔍 Checking target columns...")
        required_columns = [
            'next_day_temp', 'next_day_humidity', 'will_rain', 
            'next_day_precipitation', 'next_day_aqi', 'next_day_aqi_category'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns in dataframe: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        y_temp = df['next_day_temp']
        y_humidity = df['next_day_humidity']
        y_rain_class = df['will_rain']  # Classification: will it rain?
        y_precip_amount = df['next_day_precipitation']  # Regression: how much rain?
        y_aqi = df['next_day_aqi']
        y_aqi_category = df['next_day_aqi_category']
        
        print(f"📊 Data shapes:")
        print(f"  Features (X): {X.shape}")
        print(f"  Temp target: {y_temp.shape}")
        print(f"  Humidity target: {y_humidity.shape}")
        print(f"  Rain class target: {y_rain_class.shape}")
        print(f"  Precipitation amount target: {y_precip_amount.shape}")
        print(f"  AQI target: {y_aqi.shape}")
        print(f"  AQI category target: {y_aqi_category.shape}")
        
        # Split data ONCE for consistency - FIXED: This was the main issue
        print("\n🔪 Splitting data...")
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=42
        )
        
        # Get targets for train and test using the same indices - FIXED
        y_temp_train = y_temp.iloc[idx_train]
        y_temp_test = y_temp.iloc[idx_test]
        
        y_humidity_train = y_humidity.iloc[idx_train]  # FIXED: Now defined
        y_humidity_test = y_humidity.iloc[idx_test]    # FIXED: Now defined
        
        y_rain_class_train = y_rain_class.iloc[idx_train]
        y_rain_class_test = y_rain_class.iloc[idx_test]
        
        y_precip_amount_train = y_precip_amount.iloc[idx_train]  # FIXED: Now defined
        y_precip_amount_test = y_precip_amount.iloc[idx_test]    # FIXED: Now defined
        
        y_aqi_train = y_aqi.iloc[idx_train]
        y_aqi_test = y_aqi.iloc[idx_test]
        
        y_aqi_category_train = y_aqi_category.iloc[idx_train]
        y_aqi_category_test = y_aqi_category.iloc[idx_test]
        
        print(f"✅ Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Scale the features
        print("\n⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("🤖 Training models...")
        
        # ==================== TEMPERATURE MODEL ====================
        print("\n  🌡️ 1. Training Temperature Model...")
        temp_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        temp_model.fit(X_train_scaled, y_temp_train)
        temp_pred = temp_model.predict(X_test_scaled)
        temp_mae = mean_absolute_error(y_temp_test, temp_pred)
        temp_r2 = r2_score(y_temp_test, temp_pred)
        
        # ==================== HUMIDITY MODEL ====================
        print("  💧 2. Training Humidity Model...")
        humidity_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        humidity_model.fit(X_train_scaled, y_humidity_train)
        humidity_pred = humidity_model.predict(X_test_scaled)
        humidity_mae = mean_absolute_error(y_humidity_test, humidity_pred)
        humidity_r2 = r2_score(y_humidity_test, humidity_pred)
        
        # ==================== RAIN CLASSIFICATION MODEL ====================
        print("  🌧️ 3. Training Rain Classification Model...")
        rain_class_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        rain_class_model.fit(X_train_scaled, y_rain_class_train)
        rain_class_pred = rain_class_model.predict(X_test_scaled)
        rain_class_accuracy = accuracy_score(y_rain_class_test, rain_class_pred)
        
        # ==================== PRECIPITATION AMOUNT MODEL ====================
        print("  💦 4. Training Precipitation Amount Model...")
        precip_amount_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        
        # Train only on samples where it rains
        rain_mask_train = y_rain_class_train == 1
        rain_mask_test = y_rain_class_test == 1
        
        if rain_mask_train.sum() > 10:
            precip_amount_model.fit(X_train_scaled[rain_mask_train], y_precip_amount_train[rain_mask_train])
            if rain_mask_test.sum() > 0:
                precip_amount_pred = precip_amount_model.predict(X_test_scaled[rain_mask_test])
                precip_amount_mae = mean_absolute_error(y_precip_amount_test[rain_mask_test], precip_amount_pred)
            else:
                precip_amount_mae = 0
        else:
            print("    ⚠ Not enough raining samples for precipitation amount model")
            precip_amount_mae = 0
        
        # ==================== AQI REGRESSION MODEL ====================
        print("  🌫️ 5. Training AQI Regression Model...")
        aqi_model = RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        aqi_model.fit(X_train_scaled, y_aqi_train)
        aqi_pred = aqi_model.predict(X_test_scaled)
        aqi_mae = mean_absolute_error(y_aqi_test, aqi_pred)
        aqi_r2 = r2_score(y_aqi_test, aqi_pred)
        
        # ==================== AQI CATEGORY MODEL ====================
        print("  🏷️ 6. Training AQI Category Model...")
        aqi_category_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        aqi_category_model.fit(X_train_scaled, y_aqi_category_train)
        aqi_category_pred = aqi_category_model.predict(X_test_scaled)
        aqi_category_accuracy = accuracy_score(y_aqi_category_test, aqi_category_pred)
        
        # ==================== PRINT RESULTS ====================
        print("\n" + "="*70)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*70)
        
        print(f"\n🌡️ TEMPERATURE MODEL:")
        print(f"  R² Score: {temp_r2:.4f}")
        print(f"  MAE: {temp_mae:.2f}°C")
        
        print(f"\n💧 HUMIDITY MODEL:")
        print(f"  R² Score: {humidity_r2:.4f}")
        print(f"  MAE: {humidity_mae:.1f}%")
        
        print(f"\n🌧️ RAIN CLASSIFICATION MODEL:")
        print(f"  Accuracy: {rain_class_accuracy:.4f}")
        print(f"  Rain Rate (test set): {y_rain_class_test.mean():.2%}")
        
        if rain_mask_train.sum() > 10:
            print(f"\n💦 PRECIPITATION AMOUNT MODEL (when it rains):")
            print(f"  MAE: {precip_amount_mae:.2f} mm")
            print(f"  Trained on: {rain_mask_train.sum()} raining samples")
        
        print(f"\n🌫️ AQI REGRESSION MODEL:")
        print(f"  R² Score: {aqi_r2:.4f}")
        print(f"  MAE: {aqi_mae:.2f}")
        
        print(f"\n🏷️ AQI CATEGORY MODEL:")
        print(f"  Accuracy: {aqi_category_accuracy:.4f}")
        
        # Class distribution for AQI
        print(f"\n📊 AQI CATEGORY DISTRIBUTION (Test Set):")
        category_names = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']
        for i in range(6):
            count = (y_aqi_category_test == i).sum()
            percentage = (count / len(y_aqi_category_test)) * 100
            print(f"  {category_names[i]}: {count} samples ({percentage:.1f}%)")
        
        # ==================== SAVE MODELS ====================
        print("\n💾 Saving models...")
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(temp_model, 'models/temp_model.pkl')
        joblib.dump(humidity_model, 'models/humidity_model.pkl')
        joblib.dump(rain_class_model, 'models/rain_model.pkl')
        joblib.dump(precip_amount_model, 'models/precip_amount_model.pkl')
        joblib.dump(aqi_model, 'models/aqi_model.pkl')
        joblib.dump(aqi_category_model, 'models/aqi_category_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        print("\n" + "="*70)
        print("✅ All models saved to 'models/' directory!")
        print("="*70)
        
        return {
            'temperature': {'model': temp_model, 'r2': temp_r2, 'mae': temp_mae},
            'humidity': {'model': humidity_model, 'r2': humidity_r2, 'mae': humidity_mae},
            'rain_class': {'model': rain_class_model, 'accuracy': rain_class_accuracy},
            'precip_amount': {'model': precip_amount_model, 'mae': precip_amount_mae},
            'aqi': {'model': aqi_model, 'r2': aqi_r2, 'mae': aqi_mae},
            'aqi_category': {'model': aqi_category_model, 'accuracy': aqi_category_accuracy},
            'scaler': scaler,
            'feature_columns': feature_cols
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 WEATHER & AQI MODEL TRAINING SCRIPT")
    print("="*70)
    
    trainer = ModelTrainer()
    
    try:
        # Generate training data
        print("\n📊 Generating training data...")
        training_data = trainer.generate_training_data(5000)
        
        # Show data info
        print("\n📈 Training Data Info:")
        print(f"  Shape: {training_data.shape}")
        print(f"  Columns: {len(training_data.columns)}")
        print(f"  Sample of targets:")
        print(f"    Temperature range: {training_data['next_day_temp'].min():.1f} to {training_data['next_day_temp'].max():.1f}°C")
        print(f"    Humidity range: {training_data['next_day_humidity'].min():.1f} to {training_data['next_day_humidity'].max():.1f}%")
        print(f"    AQI range: {training_data['next_day_aqi'].min():.0f} to {training_data['next_day_aqi'].max():.0f}")
        print(f"    Rain samples: {training_data['will_rain'].sum()} ({training_data['will_rain'].mean():.2%})")
        
        # Train models
        print("\n🎯 Starting model training...")
        models = trainer.train_models(training_data)
        
        if models:
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            # Summary
            print("\n📋 MODEL SUMMARY:")
            print(f"  Temperature R²: {models['temperature']['r2']:.3f}")
            print(f"  Humidity R²: {models['humidity']['r2']:.3f}")
            print(f"  Rain Classification Accuracy: {models['rain_class']['accuracy']:.3f}")
            print(f"  AQI R²: {models['aqi']['r2']:.3f}")
            print(f"  AQI Category Accuracy: {models['aqi_category']['accuracy']:.3f}")
            
            print("\n📁 Models saved in: models/")
            print("   - temp_model.pkl")
            print("   - humidity_model.pkl")
            print("   - rain_model.pkl")
            print("   - precip_amount_model.pkl")
            print("   - aqi_model.pkl")
            print("   - aqi_category_model.pkl")
            print("   - scaler.pkl")
            
            # Test loading
            print("\n🧪 Testing model loading...")
            try:
                test_model = joblib.load('models/temp_model.pkl')
                print("  ✅ Model loading test passed!")
            except Exception as e:
                print(f"  ⚠ Model loading test failed: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()