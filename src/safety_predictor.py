import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ConstructionSafetyPredictor:
    """AI model for predicting construction site safety risks."""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def create_sample_data(self, n_samples=500):
        """Generate sample safety data"""
        data = {
            'working_height': np.random.uniform(0, 30, n_samples),
            'workers_present': np.random.randint(5, 50, n_samples),
            'equipment_count': np.random.randint(1, 10, n_samples),
            'temperature': np.random.uniform(10, 35, n_samples),
            'wind_speed': np.random.uniform(0, 25, n_samples),
            'hours_worked': np.random.uniform(1, 12, n_samples),
            'safety_measures': np.random.randint(1, 10, n_samples)
        }
        
        # Calculate risk score
        df = pd.DataFrame(data)
        df['risk_level'] = self._calculate_risk_level(df)
        return df
    
    def _calculate_risk_level(self, df):
        """Calculate risk level based on features"""
        risk_score = (
            df['working_height'] * 0.3 +
            df['workers_present'] * 0.2 +
            df['equipment_count'] * 0.15 +
            (df['temperature'] - 20).abs() * 0.1 +
            df['wind_speed'] * 0.15 +
            (df['hours_worked'] - 8).abs() * 0.1 -
            df['safety_measures'] * 0.2
        )
        return (risk_score > risk_score.median()).astype(int)
    
    def train_model(self, df):
        """Train the safety prediction model"""
        X = df.drop('risk_level', axis=1)
        y = df['risk_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        return X_test_scaled, y_test, self.model.predict(X_test_scaled)
    
    def plot_risk_factors(self):
        """Visualize importance of different risk factors"""
        # Get absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        image_path = os.path.join(project_dir, 'images', 'risk_factors.png')
        
        # Create images directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        importance = self.model.feature_importances_
        features = ['Working Height', 'Workers Present', 'Equipment Count',
                   'Temperature', 'Wind Speed', 'Hours Worked', 'Safety Measures']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.title('Risk Factors Importance in Safety Prediction')
        plt.xlabel('Importance Score')
        plt.savefig(image_path)
        plt.show()
    
    def plot_risk_distribution(self, df):
        """Plot risk distribution across different factors"""
        # Get absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        image_path = os.path.join(project_dir, 'images', 'risk_distribution.png')
        
        # Create images directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x='working_height', y='workers_present',
                       hue='risk_level', size='equipment_count')
        plt.title('Risk Distribution by Working Height and Workers Present')
        plt.savefig(image_path)
        plt.show()

def main():
    """Main function to run the predictor"""
    print("Construction Safety Risk Predictor")
    print("-" * 30)
    
    # Initialize predictor
    predictor = ConstructionSafetyPredictor()
    
    # Generate and prepare data
    print("\nGenerating safety data...")
    df = predictor.create_sample_data()
    
    # Train model
    print("Training safety prediction model...")
    X_test, y_test, y_pred = predictor.train_model(df)
    
    # Generate visualizations
    print("\nGenerating risk analysis visualizations...")
    predictor.plot_risk_factors()
    predictor.plot_risk_distribution(df)
    
    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")
    
    # Display feature importance
    importance = predictor.model.feature_importances_
    features = ['Working Height', 'Workers Present', 'Equipment Count',
               'Temperature', 'Wind Speed', 'Hours Worked', 'Safety Measures']
    
    print("\nFeature Importance:")
    for feature, imp in zip(features, importance):
        print(f"{feature}: {imp:.4f}")

if __name__ == "__main__":
    main()