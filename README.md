
# Air Quality Prediction Model

This project focuses on predicting air quality using various sensor readings. The dataset consists of sensor data collected at various times, including measurements of CO, NMHC, NOx, NO2, and several other parameters related to air quality. The goal is to build a predictive model to estimate the air quality at different points in time based on these sensor readings.

## Dataset Description

The dataset used in this project consists of sensor readings for various pollutants. The columns in the dataset include:
- **Date**: The date of the observation
- **Time**: The time at which the observation was recorded
- **CO(GT)**: Measurement of Carbon Monoxide (CO) levels
- **PT08.S1(CO)**: Sensor reading for CO levels
- **NMHC(GT)**: Measurement for non-methane hydrocarbons (NMHC)
- **NOx(GT)**: Nitrogen Oxides (NOx) levels
- **NO2(GT)**: Nitrogen Dioxide (NO2) levels
- **PT08.S2(NMHC)**: Sensor reading for NMHC levels
- **Temperature (T)**: Temperature during the observation
- **Relative Humidity (RH)**: Relative humidity during the observation
- **Absolute Humidity (AH)**: Absolute humidity during the observation

The dataset includes missing values, especially in some columns like `NMHC(GT)`, which requires handling before model training.

## Data Preprocessing

### 1. **Loading the Data**

The dataset is loaded from a CSV file using pandas `read_csv()` function. The data is delimited by semicolons (`;`).

```python
df = pd.read_csv('air_quality_data.csv', delimiter=';')
```

### 2. **Handling Missing Values**

After loading the dataset, we performed a check for missing values using `.isnull().sum()`. The dataset contains missing values for several columns, including `CO(GT)`, `PT08.S1(CO)`, and others. These missing values were handled by K-Nearest Neighbors (KNN) imputation, which was implemented using the `KNNImputer`.

```python
df.isnull().sum()
# Imputation using KNN
imputer = KNNImputer()
df_imputed = imputer.fit_transform(df)
```

### 3. **Feature Scaling**

Since the dataset involves sensor readings with different scales (e.g., CO vs. NOx), feature scaling is an essential step to normalize the data. This ensures that all features contribute equally to the model’s performance.

```python
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

## Model Selection

### 1. **Decision Tree Regressor**

A Decision Tree Regressor was chosen as one of the models for this task. It’s a popular choice for regression tasks, particularly when dealing with tabular data with non-linear relationships.

```python
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
```

#### **Feature Importance - Decision Tree**

The **Decision Tree Regressor** provides feature importance scores, which help us identify the most influential features in predicting air quality. Here is an example of the feature importance results:

```python
feature_importances = model.feature_importances_
```
The higher the importance score, the more the feature contributes to the prediction.

### 2. **Neural Network Model**

We also experimented with a Neural Network model using TensorFlow’s Keras API. The architecture consists of:
- **Input Layer**: The input layer corresponds to the features (scaled data).
- **Dense Layers**: Multiple dense layers are used for learning non-linear patterns.
- **Dropout**: Dropout layers are used for regularization to prevent overfitting.

```python
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
```

### 3. **Model Compilation & Training**

For the neural network, we used the Adam optimizer and Mean Squared Error as the loss function for regression.

```python
model.compile(optimizer=Adam(), loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

### 4. **Early Stopping**

To avoid overfitting, we employed early stopping during training. This stops the training process if the validation loss does not improve after a certain number of epochs.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
```

## Feature Importance - Permutation Method

In addition to the Decision Tree feature importance, we used the **Permutation Importance** method to assess the importance of each feature. This method shuffles a feature’s values and checks the resulting decrease in model performance.

```python
perm_importance = permutation_importance(model, X_test, y_test)
```

This method helps compare how much each feature contributes to the model’s prediction accuracy.

## Model Evaluation

After training the models, we evaluated their performance using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors.
- **R-Squared (R²)**: Indicates how well the model explains the variance in the data.

```python
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

### **Evaluation Results**

The performance evaluation results for Neural Network model is as follows:

- **Neural Network Model**:
    - MSE: 0.1279
    - MAE: 0.2323
    - R²: 0.90



## Conclusion

This project demonstrates how to predict air quality based on sensor readings using machine learning models. The combination of feature scaling, data imputation, and advanced models like neural networks allows for accurate predictions of air quality metrics.

