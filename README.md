# House Price Prediction with Linear Regression

This project implements a linear regression model to predict house prices based on key features such as square footage, bedrooms, and bathrooms, using data from the Kaggle House Prices dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Results](#results)

## Overview
In this project, we use a simple linear regression approach to model house prices using data that includes the square footage and the number of bedrooms and bathrooms in each property. This project is designed to provide insights into how linear regression can be used in real estate price predictions.

## Dataset
- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- Features used:
  - Square footage (`GrLivArea`)
  - Number of bedrooms (`BedroomAbvGr`)
  - Number of bathrooms (`FullBath`)

## Modeling Approach
The linear regression model is trained on the given features to predict the `SalePrice`. Key stages include:
1. Data Preprocessing and Cleaning
2. Feature Selection and Scaling
3. Training and Testing the Model
4. Evaluation with metrics such as Mean Absolute Error (MAE) and R-squared.

## Installation and Requirements
- Python 3.x
- Required Libraries: 
  ```sh
  pip install pandas numpy scikit-learn matplotlib seaborn
  ```

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/house-price-prediction.git
   ```
2. Download the Kaggle dataset and place it in the `data/` folder.
3. Run the main script:
   ```sh
   python house_price_prediction.py
   ```

## Results
The model provides a reasonable estimate of house prices based on the input features. Results are visualized to show the model’s accuracy across the test dataset.
