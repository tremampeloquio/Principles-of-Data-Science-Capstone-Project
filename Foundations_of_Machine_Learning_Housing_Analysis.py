#!/usr/bin/env python
# coding: utf-8

# In[61]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[3]:


#question 1

#load dataset 
data = np.genfromtxt("housingUnits.csv", delimiter=",", skip_header=1)
print("Data Shape:", data.shape) 


# In[4]:


#extract predictors
X, y = data[:, :-1], data[:, -1]

#feature names
feature_names = [
    "Median Age", "Total Rooms", "Total Bedrooms",
    "Population", "Households", "Median Income", "Proximity to Ocean"
]

#convert to df for easy visualization
df = pd.DataFrame(X, columns=feature_names)
df["House Value"] = y


# In[5]:


#histograms for total rooms, total bedrooms, population, and households
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

columns = ["Total Rooms", "Total Bedrooms", "Population", "Households"]

for i, ax in enumerate(axes.ravel()):
    ax.hist(df[columns[i]], bins=30, edgecolor="black")
    ax.set_xlabel(columns[i])
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {columns[i]}")

plt.tight_layout()
plt.show()


# In[74]:


# Create new normalized columns
df["Rooms per Population"] = df["Total Rooms"] / df["Population"]
df["Rooms per Household"] = df["Total Rooms"] / df["Households"]
df["Bedrooms per Population"] = df["Total Bedrooms"] / df["Population"]
df["Bedrooms per Household"] = df["Total Bedrooms"] / df["Households"]

# Set up the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# List of the new normalized columns
columns = ["Rooms per Population", "Rooms per Household", "Bedrooms per Population", "Bedrooms per Household"]

# Loop through and create histograms for each normalized variable
for i, ax in enumerate(axes.ravel()):
    ax.hist(df[columns[i]], bins=30, edgecolor="black")
    ax.set_xlabel(columns[i])
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {columns[i]}")

plt.tight_layout()
plt.show()


# In[59]:


# Standardized variables
rooms_per_household = df["Total Rooms"] / df["Households"]
bedrooms_per_household = df["Total Bedrooms"] / df["Households"]

rooms_per_population = df["Total Rooms"] / df["Population"]
bedrooms_per_population = df["Total Bedrooms"] / df["Population"]

# Add new standardized columns for comparison
df["Rooms per Household"] = rooms_per_household
df["Bedrooms per Household"] = bedrooms_per_household
df["Rooms per Population"] = rooms_per_population
df["Bedrooms per Population"] = bedrooms_per_population


# In[60]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plots for standardized predictors
axes[1, 0].scatter(df["Rooms per Household"], y, alpha=0.3, s=2)
axes[1, 0].set_title("Rooms per Household vs. House Value")
axes[1, 0].set_xlabel("Rooms per Household")
axes[1, 0].set_ylabel("House Value")

axes[1, 1].scatter(df["Bedrooms per Household"], y, alpha=0.3, s=2)
axes[1, 1].set_title("Bedrooms per Household vs. House Value")
axes[1, 1].set_xlabel("Bedrooms per Household")

axes[0, 0].scatter(df["Rooms per Population"], y, alpha=0.3, s=2)
axes[0, 0].set_title("Rooms per Population vs. House Value")
axes[0, 0].set_xlabel("Rooms per Population")
axes[0, 0].set_ylabel("House Value")

axes[0, 1].scatter(df["Bedrooms per Population"], y, alpha=0.3, s=2)
axes[0, 1].set_title("Bedrooms per Population vs. House Value")
axes[0, 1].set_xlabel("Bedrooms per Population")

plt.tight_layout()
plt.show()


# In[9]:


from scipy.stats import pearsonr

# Compute correlation with house value
corr_population, _ = pearsonr(df["Population"], y)
corr_households, _ = pearsonr(df["Households"], y)

print(f"Correlation (Population vs. House Value): {corr_population:.3f}")
print(f"Correlation (Households vs. House Value): {corr_households:.3f}")


# In[65]:


#question 2

# Compute standardized variables
df["Rooms per Population"] = df["Total Rooms"] / df["Population"]
df["Rooms per Household"] = df["Total Rooms"] / df["Households"]
df["Bedrooms per Population"] = df["Total Bedrooms"] / df["Population"]
df["Bedrooms per Household"] = df["Total Bedrooms"] / df["Households"]

# Compute correlations
corr_rooms_pop, _ = pearsonr(df["Rooms per Population"], df["House Value"])
corr_rooms_hh, _ = pearsonr(df["Rooms per Household"], df["House Value"])
corr_bedrooms_pop, _ = pearsonr(df["Bedrooms per Population"], df["House Value"])
corr_bedrooms_hh, _ = pearsonr(df["Bedrooms per Household"], df["House Value"])

# Print results
print(f"Correlation (Rooms per Population vs. House Value): {corr_rooms_pop:.3f}")
print(f"Correlation (Rooms per Household vs. House Value): {corr_rooms_hh:.3f}")
print(f"Correlation (Bedrooms per Population vs. House Value): {corr_bedrooms_pop:.3f}")
print(f"Correlation (Bedrooms per Household vs. House Value): {corr_bedrooms_hh:.3f}")


# In[68]:


# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # Scatter plots for each standardization method
# for i, col in enumerate(columns):
#     x = df[col]
#     y = df["House Value"]
    
#     # Fit linear regression
#     x_reshaped = x.values.reshape(-1, 1)
#     model = LinearRegression().fit(x_reshaped, y)
#     y_pred = model.predict(x_reshaped)

#     # Scatter plot with regression line
#     axes[i // 2, i % 2].scatter(x, y, alpha=0.3, s=2)
#     axes[i // 2, i % 2].plot(x, y_pred, color='red', label="Linear Fit")
#     axes[i // 2, i % 2].set_title(f"{col} vs. House Value")
#     axes[i // 2, i % 2].set_xlabel(col)
#     axes[i // 2, i % 2].set_ylabel("Median House Value")
#     axes[i // 2, i % 2].legend()

# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# for i, col in enumerate(columns):
#     x = df[col]
#     y = df["House Value"]

#     # Ensure all values are positive before applying log transformation
#     x = x + 1e-6  # Avoid log(0)
#     y = y + 1e-6

#     log_x = np.log(x)
#     log_y = np.log(y)

#     # Fit linear regression on log-transformed data
#     x_reshaped = log_x.values.reshape(-1, 1)
#     model = LinearRegression().fit(x_reshaped, log_y)
#     y_pred = model.predict(x_reshaped)

#     # Compute Pearson correlation
#     corr, _ = pearsonr(log_x, log_y)

#     # Scatter plot with regression line
#     ax = axes[i // 2, i % 2]
#     ax.scatter(log_x, log_y, alpha=0.3, s=2)
#     ax.plot(log_x, y_pred, color='red', label="Linear Fit")
#     ax.set_title(f"{col} vs. House Value (r = {corr:.2f})")
#     ax.set_xlabel(f"log({col})")
#     ax.set_ylabel("log(Median House Value)")
#     ax.legend()

# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for i, col in enumerate(columns):
    x = df[col]
    y = df["House Value"]

    # Compute Pearson correlation before any transformation
    corr, _ = pearsonr(x, y)

    # Ensure all values are positive before applying log transformation
    x_log = np.log(x + 1e-6)  # Avoid log(0)
    y_log = np.log(y + 1e-6)

    # Fit linear regression on log-transformed data for visualization
    x_reshaped = x_log.values.reshape(-1, 1)
    model = LinearRegression().fit(x_reshaped, y_log)
    y_pred = model.predict(x_reshaped)

    # Scatter plot with regression line
    ax = axes[i // 2, i % 2]
    ax.scatter(x_log, y_log, alpha=0.3, s=2)
    ax.plot(x_log, y_pred, color='red', label="Linear Fit")
    ax.set_title(f"{col} vs. House Value (r = {corr:.2f})")  # Use original correlation
    ax.set_xlabel(f"log({col})")
    ax.set_ylabel("log(Median House Value)")
    ax.legend()

plt.tight_layout()
plt.show()


# In[36]:


#question 3

# Define predictor names (with correct normalization)
predictor_columns = [
    "Median Age",
    "Rooms per Population",  # Corrected normalization
    "Bedrooms per Population",  # Corrected normalization
    "Population",
    "Households",
    "Median Income",
    "Proximity to Ocean"
]

X = df[predictor_columns]  # Predictor matrix
y = df["House Value"]  # Target variable

# Function to compute R²
def compute_r2(x, y):
    x = x.values.reshape(-1, 1)  # Reshape for sklearn
    model = LinearRegression().fit(x, y)
    return model.score(x, y)

# Compute R² for each predictor
r2_values = {col: compute_r2(df[col], y) for col in predictor_columns}

# Sort predictors by R² (descending)
r2_sorted = sorted(r2_values.items(), key=lambda x: x[1], reverse=True)

# Display results
for feature, r2 in r2_sorted:
    print(f"{feature}: R² = {r2:.3f}")

# Get most and least predictive variables
most_predictive = r2_sorted[0]
least_predictive = r2_sorted[-1]

print("\nMost Predictive Variable:", most_predictive)
print("Least Predictive Variable:", least_predictive)


# In[37]:


# Extract Median Income and House Value
x = df["Median Income"]
y = df["House Value"]

# Fit linear regression
x_reshaped = x.values.reshape(-1, 1)
model = LinearRegression().fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=2, label="Data")
plt.plot(x, y_pred, color='red', label="Linear Fit")
plt.title("Median Income vs. House Value")
plt.xlabel("Median Income ($1000s)")
plt.ylabel("Median House Value ($)")
plt.legend()
plt.show()


# In[76]:


# Extract Population and House Value
x = df["Population"]
y = df["House Value"]

# Fit linear regression
x_reshaped = x.values.reshape(-1, 1)
model = LinearRegression().fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=2, label="Data")  # Scatter plot of data points
plt.plot(x, y_pred, color='red', label="Linear Fit")  # Regression line
plt.title("Population vs. House Value")
plt.xlabel("Population")
plt.ylabel("Median House Value ($)")
plt.legend()
plt.show()


# In[38]:


#question 4

X = df[predictor_columns]  # Feature matrix
y = df["House Value"]  # Target variable

# Fit multiple linear regression
multi_model = LinearRegression().fit(X, y)
r2_full_model = multi_model.score(X, y)  # Get R² for the full model

print(f"Full Model R²: {r2_full_model:.3f}")


# In[46]:


# Fit a single-variable model using only Median Income
X_income = df[["Median Income"]]
single_model = LinearRegression().fit(X_income, y)
r2_single_model = single_model.score(X_income, y)

print(f"Best Single Predictor (Median Income) R²: {r2_single_model:.3f}")
print(f"Improvement from Single to Full Model: {r2_full_model - r2_single_model:.3f}")


# In[49]:


# Print feature coefficients
feature_importance = dict(zip(predictor_columns, multi_model.coef_))

# Sort by absolute importance
sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importance (from strongest to weakest):")
for feature, coef in sorted_features:
    print(f"{feature}: {coef:.3f}")


# In[73]:


# Define independent variables (make sure these are the correct column names)
# X = df[predictor_columns] 
# y = df["House Value"]

# # Train a multiple linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Get predictions
# y_pred = model.predict(X)

# # Scatter plot of actual vs. predicted values

# plt.figure(figsize=(6,6))
# sns.scatterplot(x=y, y=y_pred, alpha=0.3)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")  # 45-degree line
# plt.xlabel("Actual House Value")
# plt.ylabel("Predicted House Value")
# plt.title("Predicted vs. Actual House Values")
# plt.show()

from sklearn.metrics import r2_score
# Define independent variables and target
X = df[predictor_columns]  
y = df["House Value"]

# Train a multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Calculate R²
r2 = r2_score(y, y_pred)

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(6,6))
sns.scatterplot(x=y, y=y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="Linear Fit")  # 45-degree line

# Display R² value on the plot
plt.text(y.min(), y.max(), f"$R^2$ = {r2:.3f}", fontsize=12, color="blue")

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Predicted vs. Actual House Values")
plt.legend()
plt.show()


# In[50]:


#question 5

# Check collinearity between Rooms per Population & Bedrooms per Population
corr_rooms_bedrooms, _ = pearsonr(df["Rooms per Population"], df["Bedrooms per Population"])

# Check collinearity between Population & Households
corr_pop_households, _ = pearsonr(df["Population"], df["Households"])

# Print results
print(f"Correlation (Rooms per Population vs. Bedrooms per Population): {corr_rooms_bedrooms:.3f}")
print(f"Correlation (Population vs. Households): {corr_pop_households:.3f}")


# In[51]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for Rooms per Population vs. Bedrooms per Population
axes[0].scatter(df["Rooms per Population"], df["Bedrooms per Population"], alpha=0.3, s=2)
axes[0].set_title("Rooms per Population vs. Bedrooms per Population")
axes[0].set_xlabel("Rooms per Population")
axes[0].set_ylabel("Bedrooms per Population")

# Scatter plot for Population vs. Households
axes[1].scatter(df["Population"], df["Households"], alpha=0.3, s=2)
axes[1].set_title("Population vs. Households")
axes[1].set_xlabel("Population")
axes[1].set_ylabel("Households")

plt.tight_layout()
plt.show()


# In[53]:


#extra credit A

# Select predictors + outcome to check for normality
columns_to_check = [
    "Median Age",
    "Rooms per Population",
    "Bedrooms per Population",
    "Households",
    "Median Income",
    "Proximity to Ocean",
    "House Value",
    "Population"
]

# Plot histograms
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for i, col in enumerate(columns_to_check):
    axes[i].hist(df[col], bins=30, edgecolor="black")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[54]:


from scipy.stats import skew, kurtosis

# Compute skewness & kurtosis for each variable
for col in columns_to_check:
    skewness = skew(df[col])
    kurt = kurtosis(df[col])
    print(f"{col} - Skewness: {skewness:.3f}, Kurtosis: {kurt:.3f}")


# In[55]:


#extra credit b

plt.hist(df["House Value"], bins=30, edgecolor="black")
plt.title("Distribution of House Value")
plt.xlabel("Median House Value ($)")
plt.ylabel("Frequency")
plt.show()


# In[56]:


# Count how many houses have the max value
max_value_count = (df["House Value"] == df["House Value"].max()).sum()
print(f"Number of Houses at $500,000 Cap: {max_value_count}")


# In[ ]:




