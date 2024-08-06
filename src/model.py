import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import joblib

# List of top 20 features
top_20_features = [
    "Packet Length Variance",
    "Packet Length Std",
    "Avg Bwd Segment Size",
    "Max Packet Length",
    "Subflow Fwd Bytes",
    "Destination Port",
    "Bwd Packet Length Max",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Bwd Packet Length Std",
    "Total Length of Fwd Packets",
    "Subflow Bwd Bytes",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Avg Fwd Segment Size",
    "Bwd Header Length",
    "Fwd Header Length"
]

def load_and_prepare_data(input_file):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    X = df[top_20_features]
    y = df['Label']
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X, y_true):
    y_pred = (model.predict(X) > 0.5).astype("int32")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return accuracy, precision, recall, f1

class TrainingHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))

if __name__ == "__main__":
    input_file = os.path.join('..', 'data', 'processed', 'processed_data.csv') 
    X, y = load_and_prepare_data(input_file)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    nn_model = build_nn_model(input_dim=X_train.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = TrainingHistory()
    
    nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping, history])
    
    train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(nn_model, X_train, y_train)
    print(f'Training Data - Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1-Score: {train_f1}')
    
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(nn_model, X_test, y_test)
    print(f'Testing Data - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1-Score: {test_f1}')
    
    # Save the true labels and predicted probabilities for the test set
    y_test_pred_prob = nn_model.predict(X_test)
    joblib.dump((y_test, y_test_pred_prob, history.history), 'test_predictions_and_training_history.pkl')
