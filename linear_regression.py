import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load the dataset
data = pd.read_csv('./train.csv')

# Print the column names to check available features
print(data.columns)

# Select relevant features (update this line with the correct bathroom column name)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']  # Adjust 'FullBath' if necessary
target = 'SalePrice'

# Check for missing values and drop them if necessary
data = data[features + [target]].dropna()

# Define X and y
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Print model coefficients
print("Coefficients:", model.coef_)

# Visualize the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
plt.show()

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print("Cross-Validation Scores:", scores)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'.")

# Function to make predictions
def predict_price(square_footage, bedrooms, baths):
    return model.predict([[square_footage, bedrooms, baths]])

# Example usage
predicted_price = predict_price(2000, 3, 2)
print(f"Predicted price for 2000 sq ft, 3 bedrooms, and 2 baths: ${predicted_price[0]:,.2f}")
