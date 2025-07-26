import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset: Year and Closing Rank (General - Home State - Round 3)
years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
cutoffs = np.array([2194, 1965, 1686, 1965, 2080, 1944])

# Train a Linear Regression model
model = LinearRegression()
model.fit(years, cutoffs)

# Predict future cutoffs
future_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
predicted_cutoffs = model.predict(future_years)

# Plot the actual data
plt.figure(figsize=(10, 6))
plt.scatter(years, cutoffs, color='blue', label='Actual Cutoffs')

# Plot the regression line for existing data
plt.plot(years, model.predict(years), color='green', linestyle='--', label='Regression Line')

# Plot predicted points
plt.scatter(future_years, predicted_cutoffs, color='red', label='Predicted (2025–2026)')

# Annotate predicted values
for i, year in enumerate(future_years.flatten()):
    plt.annotate(f'{int(predicted_cutoffs[i]):.0f}', (year, predicted_cutoffs[i] + 30))

# Plot formatting
plt.title('MACE CSE Round 3 Cutoff Trend (2019–2024) & Prediction (2025–2026)')
plt.xlabel('Year')
plt.ylabel('Closing Rank')
plt.xticks(np.arange(2019, 2027, 1))
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
