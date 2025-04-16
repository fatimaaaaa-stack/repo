import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('emotions.csv')

# Print column names to understand the dataset
print("Column names in the dataset:")
print(data.columns)

# Preprocessing the data: Assume the last column is the label
X = data.drop(columns=['label'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
print("Training the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Extract EEG signal data directly from the dataset
eeg_signal = X.values.T  # Transpose to match the expected format (samples)
fs = 256  # Sampling frequency (Hz)

# Process wave types (Delta, Theta, Alpha, Beta, Gamma)
waves_stats = []
wave_types = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Simulate processing for each wave
danger_zone_start, danger_zone_end = 300, 500
for i, wave_type in enumerate(wave_types):
    # Use different rows of eeg_signal for each wave type as a placeholder
    wave_signal = eeg_signal[i % eeg_signal.shape[0]]  # Cycle through available channels

    # Calculate statistics for the wave
    mean_val = np.mean(wave_signal)
    max_val = np.max(wave_signal)
    min_val = np.min(wave_signal)

    # Determine sign based on mean value
    if mean_val > 0:
        sign = 'positive'
    elif mean_val < 0:
        sign = 'negative'
    else:
        sign = 'neutral'
    
    # Determine disorder and danger level based on the signal characteristics
    if mean_val > 20:
        disorder = 'stress'
        danger_level = 'Dangerous'
        measure = 'Consult a neurologist and consider medication or therapy'
    elif mean_val < -20:
        disorder = 'depression'
        danger_level = 'Dangerous'
        measure = 'Seek professional help immediately; therapy and antidepressants may be required'
    elif abs(mean_val) < 10:
        disorder = 'neutral'
        danger_level = 'Safe'
        measure = 'No action needed; maintain a healthy lifestyle'
    else:
        disorder = np.random.choice(['anxiety', 'insomnia'])
        danger_level = 'Midlevel'
        measure = 'Practice stress-reducing activities like yoga or meditation; consult a therapist if symptoms persist'

    # Store results
    waves_stats.append({
        'wave_type': wave_type,
        'mean': mean_val,
        'max': max_val,
        'min': min_val,
        'sign': sign,
        'disorder': disorder,
        'danger_level': danger_level,
        'recommended_measure': measure
    })
    # Plot the original EEG signal
    plt.figure(figsize=(10, 6))
    plt.plot(wave_signal, label=f'{wave_type} Wave - Original')
    plt.title(f"{wave_type} Wave - Original Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Plot the EEG signal with a highlighted danger zone
    plt.figure(figsize=(10, 6))
    plt.plot(wave_signal, label=f'{wave_type} Wave')
    plt.axvspan(danger_zone_start, danger_zone_end, color='red', alpha=0.3, label='Danger Zone')
    plt.title(f"{wave_type} Wave - Signal with Danger Zone Highlighted")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Simulate predicted EEG signal for comparison
    predicted_signal = wave_signal * np.random.uniform(0.9, 1.1, wave_signal.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_signal, label=f'{wave_type} Wave - Predicted', color='green')
    plt.title(f"{wave_type} Wave - Predicted Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Plot comparison between original and predicted EEG signals
    plt.figure(figsize=(10, 6))
    plt.plot(wave_signal, label=f'{wave_type} Wave - Original')
    plt.plot(predicted_signal, label=f'{wave_type} Wave - Predicted', linestyle='dashed', color='orange')
    plt.title(f"{wave_type} Wave - Comparison of Original and Predicted Signals")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


# Generate additional data rows as before
num_additional_rows = np.random.randint(1500, 2000) - len(waves_stats)

additional_stats = []
for _ in range(num_additional_rows):
    random_wave = np.random.choice(wave_types)
    random_mean = np.random.uniform(-50, 50)
    random_max = random_mean + np.random.uniform(0, 20)
    random_min = random_mean - np.random.uniform(0, 20)

    if random_mean > 20:
        danger_level = 'Dangerous'
        measure = 'Consult a neurologist and consider medication or therapy'
        disorder = 'stress'
    elif random_mean < -20:
        danger_level = 'Dangerous'
        measure = 'Seek professional help immediately; therapy and antidepressants may be required'
        disorder = 'depression'
    elif abs(random_mean) < 10:
        danger_level = 'Safe'
        measure = 'No action needed; maintain a healthy lifestyle'
        disorder = 'neutral'
    else:
        danger_level = 'Midlevel'
        measure = 'Practice stress-reducing activities like yoga or meditation; consult a therapist if symptoms persist'
        disorder = np.random.choice(['anxiety', 'insomnia'])

    additional_stats.append({
        'wave_type': random_wave,
        'mean': random_mean,
        'max': random_max,
        'min': random_min,
        'sign': 'positive' if random_mean > 0 else 'negative',
        'disorder': disorder,
        'danger_level': danger_level,
        'recommended_measure': measure
    })

# Combine original and additional stats
waves_stats.extend(additional_stats)

# Convert the list to a DataFrame
waves_df = pd.DataFrame(waves_stats)

# Save processed data to a CSV file
waves_df.to_csv('progress.csv', index=False)

# Print summary of processed data
print("Processed Wave Data:")
print(waves_df.head())
