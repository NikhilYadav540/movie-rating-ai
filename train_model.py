import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. LOAD YOUR LOCAL FILE ---
print("📂 Loading local dataset...")
try:
    # Make sure 'IMDb Movies India.csv' is in the same folder!
    df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
    print(f"✅ Loaded {len(df)} raw movies!")
except FileNotFoundError:
    print("❌ File not found! Please download 'IMDb Movies India.csv' and put it in this folder.")
    exit()

# --- 2. CLEANING THE DATA (Fixed for messy symbols) ---
print("🧹 Cleaning data...")

# FIX 1: Clean 'Duration'
# Remove ' min', turn to string first to avoid errors, then coerce errors
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# FIX 2: Clean 'Votes'
# This was causing your error. We remove commas, then Force Convert to numbers.
# If it finds '$5.16M', it becomes NaN (Blank) automatically.
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# FIX 3: Clean 'Rating'
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop all rows that became NaN (Blanks) because of bad data
df.dropna(subset=['Duration', 'Votes', 'Rating'], inplace=True)

# Create Dummy Budget (Engineering Hack)
df['Budget'] = df['Votes'] * 1000 

print(f"✅ Training on {len(df)} CLEAN Indian movies!")

# Features (Inputs)
X = df[['Budget', 'Duration', 'Votes']]
y = df['Rating']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, 'scaler.pkl')

# --- 3. BUILD MODEL ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)), 
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- 4. TRAIN ---
print("🧠 Training on Indian Data...")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

loss = model.evaluate(X_test_scaled, y_test)
print(f"📉 Final Error: {loss:.4f}")

model.save('movie_rating_model.h5')
print("🎉 Model trained on INDIAN DATA! Run app.py now.")