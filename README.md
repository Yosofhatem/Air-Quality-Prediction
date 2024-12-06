# Air Quality Prediction using Neural Networks

This project is focused on predicting air quality based on environmental factors using a neural network model. The goal is to estimate pollution levels (e.g., CO(GT)) from various sensors and environmental parameters such as temperature, humidity, and other pollutants.

The model was trained using the dataset, evaluated, and fine-tuned to achieve optimal performance. Various techniques, such as feature importance analysis and hyperparameter tuning, were used to improve the model's accuracy.

---

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Model Architecture](#model-architecture)
3. [Model Evaluation](#model-evaluation)
4. [Predictions vs Actual Values](#predictions-vs-actual-values)
5. [Model Performance Metrics](#model-performance-metrics)
6. [Feature Importance Analysis](#feature-importance-analysis)
7. [Visualizing Weights and Biases](#visualizing-weights-and-biases)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Streamlit Deployment](#streamlit-deployment)
---

# Data Preprocessing

Before training the model, the data is preprocessed by performing exploratory data analysis (EDA), handling missing values, handling duplicates, converting data types, and managing outliers. Here are the detailed steps:

## Exploratory Data Analysis (EDA)

```python
# Drop unnecessary columns 'Unnamed: 15' and 'Unnamed: 16' from the DataFrame
df.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)

# Display the first 5 rows of the DataFrame
df.head()

# Check the dimensions of the DataFrame
df.shape

# Display summary information about the DataFrame, including column data types and non-null counts
df.info()

# Get summary statistics
df.describe().T
```

**Explanation**:
- The unnecessary columns ('Unnamed: 15', 'Unnamed: 16', Date, and Time) are removed.
- Summary statistics, non-null counts, and data types are explored for the remaining columns.

## Handling Null Values

```python
# Check for missing values
df.isnull().sum()

# Dropping rows with missing values (NaN)
df.dropna(axis=0, inplace=True)
```

## Handling Duplicate Values

```python
# Count the number of duplicate rows in the DataFrame
df.duplicated().sum()

# Remove duplicate rows from the DataFrame
df.drop_duplicates(inplace=True, ignore_index=True)
```

## Convert Data Types

```python
# Convert columns with commas as decimals to float type
columns_to_convert = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']

for column_to_convert in columns_to_convert:
    # Replace commas with dots and then convert to float
    df[column_to_convert] = df[column_to_convert].str.replace(',', '.').astype(float)

# Display summary information about the DataFrame, including column data types and non-null counts
df.info()
```

## Handling Malignant Null Values

```python
# Iterate over each column and replace -200.0 with NaN
for col in df.columns:
    df[col] = df[col].replace(-200.0, np.nan)

# Check for missing values again
df.isnull().sum()

# Calculate the percentage of missing values for each column
df.isnull().sum() / len(df) * 100
```

## Handle Missing Values Using KNN Imputation

```python
from sklearn.impute import KNNImputer

# Handle missing values using KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Display summary information about the DataFrame
df_imputed.info()
```

## Show Outliers Using Boxplot

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot to detect outliers
plt.figure(figsize=(20, 8))
sns.boxplot(data=df_imputed)

# Add title
plt.title("Boxplot for Outliers", fontsize=16)

# Display the plot
plt.show()
```

## Handling Outliers

```python
# Calculate Z-scores for all columns
z_scores = (df_imputed - df_imputed.mean()) / df_imputed.std()

# Set a threshold for Z-scores (3)
threshold = 3

# Filter out rows where any column has a Z-score greater than the threshold
df_no_outliers = df_imputed[(z_scores.abs() <= threshold).all(axis=1)]

df_imputed = df_no_outliers

# Create a boxplot again after removing outliers
plt.figure(figsize=(20, 8))
sns.boxplot(data=df_imputed)

# Add title
plt.title("Boxplot for Outliers", fontsize=16)

# Display the plot
plt.show()
```

## Final Preprocessed Data

```python
# After handling missing values and outliers, display final data shape
df_imputed.shape
```


**Conclusion:**

The preprocessing steps ensure the model receives scaled input features, which is important for neural networks, and we split the data into training and testing sets to evaluate model performance accurately.

---

## Model Architecture

The neural network model consists of dense layers with ReLU activation, followed by dropout layers to prevent overfitting.

```python
# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
```

**Conclusion:**

The architecture consists of two hidden layers with dropout applied after each, helping to prevent overfitting and improving generalization.

---

## Model Evaluation

We begin by evaluating the model on the test set and plotting the loss over the epochs for both training and validation sets.

```python
# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss}')

# Plot training history (loss over epochs)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Highlight the point where training stopped
plt.axvline(x=early_stopping.stopped_epoch, color='r', linestyle='--', label=f'Best Epoch {early_stopping.stopped_epoch}')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Conclusion from Visualization:**

From the plot, we can observe the behavior of the loss function throughout the epochs. The training and validation losses seem to converge, indicating that the model is learning effectively. The point where training stopped (marked by the red line) shows that the model achieved its best performance before overfitting.

---

## Predictions vs Actual Values

After training, the model makes predictions on the test set. Here, we visualize the predicted values versus the actual values to understand the accuracy of our predictions.

```python
# Predict using the trained model
y_pred = model.predict(X_test)

# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
plt.title('Actual vs Predicted CO(GT) Levels')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
```

**Conclusion from Visualization:**

The scatter plot shows the correlation between actual and predicted values. The red dashed line represents a perfect prediction, and the points scattered around this line indicate how well the model predicted the CO(GT) levels.

---

## Model Performance Metrics

To evaluate the model's performance, we compute the following metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.

```python
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')
```

**Results:**

- **Mean Squared Error (MSE):** 0.1279025925033338
- **Mean Absolute Error (MAE):** 0.23225950731224274
- **R² Score:** 0.9082963043707339

**Conclusion from Metrics:**

The MSE is relatively low, indicating small errors in predictions. The MAE further supports this, showing that on average, the model's predictions are close to the actual values. The R² score of 0.91 indicates that the model explains 91% of the variance in the target variable, which is a strong result for a regression task.

---

## Feature Importance Analysis

We conducted feature importance analysis using two methods: Permutation Importance and Decision Tree Regressor.

### Permutation Importance

```python
# Compute Permutation Importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

# Extract the importance values
importance_values = result.importances_mean
std_dev = result.importances_std

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_values,
    'Standard Deviation': std_dev
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], xerr=feature_importance_df['Standard Deviation'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance using Permutation Importance')
```

**Conclusion from Permutation Importance:**

The feature importance plot shows which features have the most impact on the model's predictions. Features like C6H6(GT) and PT08.S2(NMHC) have high importance, while others have less influence.

### Decision Tree for Feature Importance

```python
# Initialize the decision tree regressor
regressor = DecisionTreeRegressor()

# Train the regressor
regressor.fit(X_train, y_train)

# Get feature importances
feature_importances = regressor.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(feature_importance_df)
```

**Results:**

| Feature          | Importance |
|------------------|------------|
| C6H6(GT)         | 0.690663   |
| PT08.S2(NMHC)    | 0.134098   |
| NOx(GT)          | 0.046285   |
| NO2(GT)          | 0.038337   |
| PT08.S1(CO)      | 0.019748   |

**Conclusion from Decision Tree:**

The decision tree provides a similar view of feature importance, confirming that C6H6(GT) is the most important feature, followed by PT08.S2(NMHC).

---

## Visualizing Weights and Biases of Dense Layers

We also visualize the distribution of weights and biases for each dense layer in the neural network.

```python
# Initialize a counter for Dense layers
dense_layer_idx = 0

for idx, layer in enumerate(model.layers):
    if isinstance(layer, Dense):  # Only process Dense layers
        print(f"Accessing weights and biases for Dense layer {dense_layer_idx}: {layer.name}")
        
        try:
            weights, biases = layer.get_weights()  # Get weights and biases for Dense layer
            
            # Visualizing the weights
            plt.figure(figsize=(10, 6))
            plt.hist(weights.flatten(), bins=50)
            plt.title(f'Weights Distribution in Dense Layer {layer.name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()

            # Visualizing the biases
            plt.figure(figsize=(10, 6))
            plt.hist(biases.flatten(), bins=50)
            plt.title(f'Biases Distribution in Dense Layer {layer.name}')
            plt.xlabel('Bias Value')
            plt.ylabel('Frequency')
            plt.show()

            # Increment the counter for Dense layers
            dense_layer_idx += 1

        except ValueError:
            print(f"Layer {layer.name} does not have weights or biases.")
```

**Conclusion from visualization of Weights and Biases:**

The histograms of weights and biases show their distribution across the model layers. This helps in understanding how the model assigns importance to different parameters.

---

## Hyperparameter Tuning

We perform hyperparameter tuning to optimize the model's performance, using a grid search over combinations of learning rates, dropout rates, and other parameters.

```python
# Define the model-building function with tunable hyperparameters
def build_model(learning_rate=0.001, dropout_rate_1=0.2, dropout_rate_2=0.3, neurons_1=128, neurons_2=64):
    model = Sequential()

    # Define input layer
    model.add(Input(shape=(X_train.shape[1],)))  

    # First hidden layer
    model.add(Dense(neurons_1, activation='relu'))

    # Dropout layer
    model.add(Dropout(dropout_rate_1))

    # Second hidden layer
    model.add(Dense(neurons_2, activation='relu'))

    # Dropout layer
    model.add(Dropout(dropout_rate_2))

    # Output layer (single neuron for regression)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mse'])

    return model

# Hyperparameter search
results = []

# Manually search through combinations of hyperparameters
for lr in learning_rates:
    # Code continues for each hyperparameter combination...

```

**Best Hyperparameters:**

```python
lr=0.001, dr1=0.2, dr2=0.3, n1=128, n2=64, batch_size=32, epochs=50
Best MSE: 0.128878663132922
```

**Conclusion from Hyperparameter Tuning:**

The hyperparameter search resulted in a combination that minimized the MSE, leading to better model performance.

---

## Streamlit Deployment

## Overview
This app predicts the Carbon Monoxide (CO) concentration in the air based on various environmental parameters such as temperature, humidity, and pollutant concentrations. It uses a pre-trained machine learning model to generate predictions and classifies the results into different health risk categories.

## Features
- **Interactive Input**: Users can enter values for environmental features like temperature, humidity, and pollutant concentrations.
- **CO Prediction**: Predicts CO levels in mg/m³ based on user inputs.
- **Health Risk Classification**: Categorizes predicted CO levels into health risk levels (Normal, Cautionary, Moderate, Hazardous, Dangerous, Life-Threatening).
- **Real-time Feedback**: Provides actionable recommendations based on the CO level category.

## Deployment

### Requirements
- **Python**: Version 3.7 or above
- **Libraries**:
  - TensorFlow
  - Pandas
  - NumPy
  - Streamlit
  - Joblib

Install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```text
tensorflow
pandas
numpy
streamlit
joblib
```

### App Files
- `model.keras`: Pre-trained Keras model for CO prediction.
- `scaler.pkl`: Pre-fitted scaler for preprocessing input data.
- `app.py`: Main Streamlit application file.

### How to Run the App
1. Clone the repository or download the app files.
2. Ensure that the required dependencies are installed using `pip install -r requirements.txt`.
3. Place the pre-trained model (`model.keras`) and scaler file (`scaler.pkl`) in the same directory as `app.py`.
4. Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
5. Open the app in your default web browser or use the provided local URL (e.g., http://localhost:8501).

## Usage
1. Open the app in your browser.
2. Enter values for the following environmental parameters:
   - Temperature (T)
   - Relative Humidity (RH)
   - NO2 Concentration (NO2(GT))
   - CO Concentration (PT08.S1(CO))
   - NMHC Concentration (NMHC(GT))
   - Benzene Concentration (C6H6(GT))
   - NOx Concentration (NOx(GT))
   - Additional pollutants and parameters
3. Click **Make Prediction**.
4. View the predicted CO level (in mg/m³), its health risk classification, and the corresponding health message.

## Health Risk Categories
- **Normal**: CO levels are within safe limits. No immediate action required.
- **Cautionary**: Slightly elevated CO levels. Monitor the air quality.
- **Moderate**: Moderate CO levels. Prolonged exposure could be harmful.
- **Hazardous**: Hazardous CO levels detected. Immediate action required.
- **Dangerous**: Dangerous CO levels detected. Evacuate and seek medical attention.
- **Life-Threatening**: Life-threatening CO levels detected. Immediate evacuation required.

## Error Handling
- Displays an error if `model.keras` or `scaler.pkl` is not found.
- Alerts the user if there is an issue during prediction.

## Code Explanation
### Loading the Model and Scaler
The app loads the pre-trained machine learning model and the scaler:
```python
scaler = joblib.load('scaler.pkl')  # Load the pre-fitted scaler
model = load_model('model.keras')  # Load the pre-trained Keras model
```

### User Input and Data Preprocessing
Inputs from the user are collected via Streamlit’s `number_input` widget. The data is then scaled:
```python
inputs = {}
for col in columns:
    inputs[col] = st.number_input(f"Enter value for {col}", value=0.0000, format="%.4f")
df_test = pd.DataFrame([inputs])
df_scaled = scaler.transform(df_test)  # Scale the input data
```

### Prediction and Classification
The scaled data is used to predict the CO level. Predictions are clipped to predefined limits, and the CO level is categorized:
```python
predictions = model.predict(df_scaled)
predicted_value = predictions[0][0]
limited_prediction = np.clip(predicted_value, MIN_CO, MAX_CO)
co_category = classify_co_level(limited_prediction)
```

### Displaying Results
The app displays the prediction and corresponding health messages based on the CO category:
```python
st.write(f"Prediction For CO(GT): {limited_prediction:.6f} mg/m³")
st.write(f"CO Level Category: {co_category}")
if co_category == "Normal":
    st.success("The CO level is within safe limits. No immediate action required. 🛡️")
elif co_category == "Cautionary":
    st.warning("CO levels are slightly elevated. Monitor the air quality. ⚠️")
# Additional cases omitted for brevity
```

## Example Interaction
- Enter realistic values for the environmental parameters.
- Click **Make Prediction** to receive:
  - Predicted CO level (mg/m³).
  - Health risk category.
  - Recommended actions based on the category.

## Future Enhancements
- Input validation to ensure realistic and consistent user inputs.
- Display model version or training date for transparency.
- Unit conversion options for temperature and pollutant measurements.
- Progress indicators for better user experience during computation.

