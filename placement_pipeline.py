
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


#reproduce identify leakage
#task1

# Task 1: Demonstrate data leakage

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X, y = make_classification(n_samples=1000, n_features=10, random_state=42)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)

#task2

# Task 2: Fix using Pipeline and Cross-Validation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("CV Mean Accuracy:", cv_scores.mean())
print("CV Std Dev      :", cv_scores.std())
#task3

# Task 3: Compare different tree depths

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

depths = [1, 5, 20]
results = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    results.append({
        "max_depth": d,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })

df_results = pd.DataFrame(results)
print(df_results)