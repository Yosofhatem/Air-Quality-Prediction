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

---

## Data Preprocessing

Before training the model, the data is preprocessed by normalizing the features and splitting the data into training and testing sets.

```python
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
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

