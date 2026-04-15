
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
np.random.seed(42)
n = 50
area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

price_lakhs = (
    0.05 * area_sqft +
    5 * num_bedrooms -
    0.8 * age_years +
    np.random.normal(0, 10, n)
)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

y_pred = model.predict(X_test)
print("\nActual vs Predicted (first 5):")
for actual, pred in list(zip(y_test, y_pred))[:5]:
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

    #task2
    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

#task3
import matplotlib.pyplot as plt

# Residuals
residuals = y_test - y_pred

# Plot histogram
plt.figure()
plt.hist(residuals, bins=10)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()