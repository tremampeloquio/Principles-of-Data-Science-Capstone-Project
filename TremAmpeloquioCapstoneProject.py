#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:59:19 2024

@author: tremampeloquio
"""
N_number = 10691537

#import data files
import pandas as pd

quantitative_df = pd.read_csv('rmpCapstoneNum.csv')
qualitative_df = pd.read_csv('rmpCapstoneQual.csv')

#%%
#cleaning

#filter quant df for professors with more than the median number of ratings

valid_indices = quantitative_df[quantitative_df.iloc[:,2].notna()].index
quantitative_df = quantitative_df.iloc[valid_indices]

#average number of ratings:
avg_num_ratings = sum(quantitative_df.iloc[:,2])/len(quantitative_df.iloc[:,2])

#analyze distribution of number of ratings
print(quantitative_df.iloc[:,2].describe())

###half of the number of ratings are below 3/half are above 3

mean_number_of_ratings = avg_num_ratings
median_number_of_ratings = quantitative_df.iloc[:,2].median()

###the mean is greater than the median so the number of ratings distribution is slightly skewed right

quant_df_with_threshold = quantitative_df[quantitative_df.iloc[:,2] >= median_number_of_ratings]

print(quant_df_with_threshold.describe())

###consider matching the rows of the quant_df with that of the qual_df

#%%
#set alpha for significance testing
alpha = 0.005
#%%
#Q1

#H0: no diff b/t (median) male and female professor ratings
#H1: (median) rating for male professors is higher than that for female professors

#descriptive stats for each gender

#get number of ambiguous entries
ambiguous_gender_rows = quant_df_with_threshold[(quant_df_with_threshold.iloc[:,6] == 1) & (quant_df_with_threshold.iloc[:,7] == 1)]
print(len(ambiguous_gender_rows))

#get percentage of ambiguous entries
print(round(len(ambiguous_gender_rows)/len(quant_df_with_threshold), 4))

#remove ambiguous entries
quant_df_with_threshold = quant_df_with_threshold[~((quant_df_with_threshold.iloc[:,6] == 1) & (quant_df_with_threshold.iloc[:,7] == 1))]

#get number of male entries
male_rows = quant_df_with_threshold[quant_df_with_threshold.iloc[:,6] == 1]
male_count = quant_df_with_threshold[quant_df_with_threshold.iloc[:,6] == 1].shape[0]

#get number of female entries
female_rows = quant_df_with_threshold[quant_df_with_threshold.iloc[:,7] == 1]
female_count = quant_df_with_threshold[quant_df_with_threshold.iloc[:,7] == 1].shape[0]

#checking if variances are equal
from scipy.stats import levene

Q1_df = quant_df_with_threshold
male_ratings = male_rows.iloc[:,0]
female_ratings = female_rows.iloc[:,0]

stat, p_value = levene(male_ratings, female_ratings)
if p_value < alpha:
    print("Variances are significantly different. Use Welch's t-test.")
else:
    print("Variances are approximately equal. Use Student's t-test.")
    
#checking normality of male/female ratings to see if t-test is feasible
from scipy.stats import kstest

# test for male ratings
ks_stat_male, ks_p_male = kstest(male_ratings, 'norm')
print(f"KS Test for Male Ratings: Stat={ks_stat_male}, P-value={ks_p_male}")

# test for female ratings
ks_stat_female, ks_p_female = kstest(female_ratings, 'norm')
print(f"KS Test for Female Ratings: Stat={ks_stat_female}, P-value={ks_p_female}")

#distributions of male/female ratings are not normal, so turning to mann-whitney U test
from scipy.stats import mannwhitneyu

# Perform the Mann-Whitney U test
u_stat, p_value = mannwhitneyu(male_ratings, female_ratings, alternative='greater')

print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"P-value: {p_value}")

import matplotlib.pyplot as plt

#visualize figure
boxplots = plt.boxplot([male_ratings, female_ratings],
            labels=['Male', 'Female'],
            widths=0.6,
            patch_artist=True)

colors = ['lightblue', 'lightpink']  # Colors for male and female boxplots
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)
    
medians = boxplots['medians']  # Median lines
for i, median in enumerate(medians):
    median_value = median.get_ydata()[0]  # Extract the median value
    plt.text(
        i + 1,  # x-coordinate (boxplot index starts at 1)
        median_value,  # y-coordinate
        f'{median_value:.2f}',  # Format the label
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        color='black',  # Label color
        fontsize=10  # Font size
    )
    
plt.ylabel('Ratings')
plt.title('Ratings by Gender')
plt.show()

#calculating effect size
n1 = len(male_ratings)
n2 = len(female_ratings)
effect_size = 2 * u_stat / (n1 * n2) - 1
print(f"Effect Size (Rank-Biserial Correlation): {effect_size}")

#%%
#giving labels to the columns so its easier to extract

labels = ['average rating', 'average difficulty', 'number of ratings', 'received a pepper?', 
          'proportion of students that would take class again', 'number of ratings from online classes',
          'male gender', 'female gender']

quant_df_labels = quant_df_with_threshold
quant_df_labels.columns = labels
#%%
#Q2
import numpy as np

Q2_df = quant_df_labels

# Calculate the 99th percentile of the 'number of ratings' column
percentile_99 = np.percentile(Q2_df['number of ratings'], 99)

# Filter the data to include only rows where 'number of ratings' <= 99th percentile
Q2_df_filtered = Q2_df[Q2_df['number of ratings'] <= percentile_99]



#operationalize experience as number of ratings
#operationalize quality with average rating

#visualize relationship b/t experience and quality
import seaborn as sns

# Scatterplot with a trendline
#sns.regplot(data=Q2_df, x='number of ratings', y='average rating', scatter_kws={'color': 'orange'}, line_kws={'color': 'blue'})

# Add labels and title
#plt.xlabel('Number of Ratings (Experience)')
#plt.ylabel('Average Rating (Quality)')
#plt.title('Experience vs Quality of Teaching')

# Set y-axis limits
#plt.ylim(0, 5)

# Show the plot
#plt.show()

#checking linear regression
import statsmodels.api as sm

# Perform linear regression on the filtered data
X_filtered = Q2_df_filtered['number of ratings']  # Experience predictor
y_filtered = Q2_df_filtered['average rating']  # Quality target

#X = Q2_df['number of ratings'] # Create a regression model (Experience predicting Quality)
#y = Q2_df['average rating']
X_filtered = sm.add_constant(X_filtered)  # Add a constant (intercept) to the model
#X = sm.add_constant(X) # Add a constant (intercept) to the model

model_filtered = sm.OLS(y_filtered, X_filtered).fit()  # Fit the regression model

#model = sm.OLS(y, X).fit()  # Fit the regression model

print(model_filtered.summary())

# Visualize the filtered data
sns.regplot(data=Q2_df_filtered, x='number of ratings', y='average rating', 
            scatter_kws={'color': 'orange'}, line_kws={'color': 'blue'})

plt.xlabel('Number of Ratings (Experience)')
plt.ylabel('Average Rating (Quality)')
plt.title('Experience vs Quality of Teaching (99th Percentile Filtered)')
#plt.ylim(0, 5)
plt.show()

#print(model.summary()) # Print the summary of the model


#checking Spearman Rank correlation

#H0: no monotonic association b/t experience and quality
#H1: monotonic association b/t experience and quality
from scipy.stats import spearmanr

experience = Q2_df['number of ratings']
quality = Q2_df['average rating']

rho, p_value = spearmanr(experience, quality)

print(f"Spearman correlation coefficient (rho): {rho:.4f}")
print(f"P-value: {p_value:.4f}")

#checking Pearson correlation

#H0: no linear relat. b/t variables
#H1: linear relat. b/t variables
from scipy.stats import pearsonr

corr, p_value = pearsonr(Q2_df['number of ratings'], Q2_df['average rating'])
print(f"Pearson correlation: {corr}")
print((f"P-Value: {p_value}"))



#%%
#Q3 redo

# Drop rows with NaN in the relevant columns
filtered_df = quant_df_labels.dropna(subset=['average rating', 'average difficulty'])

# Extract the columns for correlation
ratings = filtered_df['average rating']
difficulty = filtered_df['average difficulty']

# Calculate Pearson correlation coefficient
correlation_matrix = np.corrcoef(ratings, difficulty)
correlation_coefficient = correlation_matrix[0, 1]

print("Pearson Correlation Coefficient:", correlation_coefficient)

# Scatterplot for filtered data
sns.scatterplot(data=filtered_df, 
                x='average rating', 
                y='average difficulty')

plt.xlabel('Average Rating')
plt.ylabel('Average Difficulty')
plt.title('Relationship Between Rating and Average Difficulty')
plt.show()


#%%
#Q3

Q3_df = quant_df_labels


#checking if x/y variable is normally distributed with KS
#H0: normal distribution
#H1: non normal distribution
ks_stat_avg_diff, ks_p_avg_diff = kstest(Q3_df['average difficulty'], 'norm')
print(f"KS Test for Avg Diff: Stat={ks_stat_avg_diff}, P-value={ks_p_avg_diff}")

ks_stat_avg_rating, ks_p_avg_rating = kstest(Q3_df['average rating'], 'norm')
print(f"KS Test for Avg Rating: Stat={ks_stat_avg_rating}, P-value={ks_p_avg_rating}")

#since there's no discernable linear relat. & data non normal --> Spearman
#H0: no monotonic relat. b/t avg diff & avg rating
#H1: monotonic relat. b/t avg diff & avg rating
spearman_corr, spearman_p_val = spearmanr(Q3_df['average difficulty'], 
                                          Q3_df['average rating'])

print(f"Spearman correlation coefficient: {spearman_corr:}")
print(f"P-value: {spearman_p_val:}")

#scatterplot with spearman correlation
sns.regplot(data=Q3_df, x='average difficulty', y='average rating', 
            lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Relationship between Difficulty and Rating')

plt.text(0.05, 0.9, f"Spearman Corr: {spearman_corr:.2f}\n(p = {spearman_p_val:.2e})", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

#%%
#renaming the dataframe for simplicity

df = quant_df_labels
#%%
#Q4

#operationalize: "a lot" --> custom threshold

#finding median # ratings from online classes
median_online_classes = df['number of ratings from online classes'].median()
print(median_online_classes)

#finding 75th percentile
Q3_online_classes = df['number of ratings from online classes'].quantile(0.75)
print(Q3_online_classes)

df_high_online = df[df['number of ratings from online classes'] > 2]
df_low_online = df[df['number of ratings from online classes'] <= 2]

#checking if variances are equal
levene_stat, levene_p_value = levene(df_high_online['average rating'],
                    df_low_online['average rating'])
if levene_p_value < alpha:
    print("Variances are significantly different. Use Welch's t-test.")
else:
    print("Variances are approximately equal. Use Student's t-test.")

#H0: no diff b/t professors who teach many online classes and those who teach fewer/none
#H1: professors who teach many online classes have higher ratings than those who teach fewer/none

import scipy.stats as stats

#mann whitney u
ratings_high_online = df_high_online['average rating']
ratings_low_online = df_low_online['average rating']
stat, p_value = stats.mannwhitneyu(ratings_high_online, ratings_low_online, alternative='greater')
print(f"U-statistic: {stat}, p-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: Online professors have significantly higher ratings.")
else:
    print("Fail to reject the null hypothesis: No significant difference in ratings.")


#H0: no diff b/t professors who teach many online classes and those who teach fewer/none
#H1: professors who teach many online classes have lower ratings than those who teach fewer/none
stat, p_value = stats.mannwhitneyu(ratings_high_online, ratings_low_online, alternative='less')
print(f"U-statistic: {stat}, p-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: Online professors have significantly lower ratings.")
else:
    print("Fail to reject the null hypothesis: No significant difference in ratings.")


# Plot the boxplot

# Visualize figure
boxplots = plt.boxplot(
    [ratings_high_online, ratings_low_online],
    labels=['High Online', 'Low Online'],
    widths=0.6,
    patch_artist=True
)

# Define colors for the boxplots
colors = ['lightgreen', 'lightblue']  # Colors for High Online and Low Online boxplots
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Annotate medians
medians = boxplots['medians']  # Median lines
for i, median in enumerate(medians):
    median_value = median.get_ydata()[0]  # Extract the median value
    plt.text(
        i + 1,  # x-coordinate (boxplot index starts at 1)
        median_value,  # y-coordinate
        f'{median_value:.2f}',  # Format the label
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        color='black',  # Label color
        fontsize=10  # Font size
    )

# Add labels and title
plt.ylabel('Average Rating')
plt.title('Ratings by Mode of Professor (Mostly Online or Not)')
plt.show()


#find effect size here!
# Calculate the sample sizes
n1 = len(ratings_high_online)  # Number of professors with high online ratings
n2 = len(ratings_low_online)   # Number of professors with low online ratings
r_biserial = (2 * stat) / (n1 * n2) - 1
print(f"Rank Biserial Correlation: {r_biserial}")


#%%
#Q5

filtered_df = df.dropna(subset=['average rating', 'proportion of students that would take class again'])
print("Original number of rows:", len(df))
print("Filtered number of rows:", len(filtered_df))

ratings = filtered_df['average rating']
proportion = filtered_df['proportion of students that would take class again']

#checking for normality
from scipy.stats import shapiro
shapiro_ratings = shapiro(ratings)
shapiro_proportion = shapiro(proportion)

print(f"Ratings normality: W={shapiro_ratings.statistic}, p={shapiro_ratings.pvalue}")
print(f"Proportion normality: W={shapiro_proportion.statistic}, p={shapiro_proportion.pvalue}")

#not normally distributed so use Spearman
#get correlation coefficient

rho, p_value = spearmanr(ratings, proportion)

print(f"Spearman correlation coefficient (rho): {rho:.4f}")
print(f"P-value: {p_value:.4f}")

#make scatterplot with spearman trendline
sns.scatterplot(x=ratings, y=proportion, color='orange')
sns.regplot(data=filtered_df, x='average rating', y='proportion of students that would take class again', 
            scatter_kws={'color': 'orange'}, line_kws={'color': 'blue'}, lowess=True)
plt.xlabel('Average Rating')
plt.ylabel('Proportion of Students Taking Class Again')
plt.title("Spearman Correlation Trendline")

plt.show()
#%%
#Q6

hot = df[df['received a pepper?'] == 1]
hot['average rating'].describe()
hot_ratings = hot['average rating']

not_hot = df[df['received a pepper?'] == 0]
not_hot['average rating'].describe()
not_hot_ratings = not_hot['average rating']

#plot historgrams
sns.histplot(hot_ratings, kde=True, color='red', label='Hot Professors')
sns.histplot(not_hot_ratings, kde=True, color='blue', label='Not Hot Professors')
plt.xlabel('Average Rating')
plt.title('Distribution of Ratings')
plt.legend()
plt.show()

#data not normal so doing mann whitney U

#H0: 'hot'/non 'hot' professors have same rating
#H1: 'hot' professors have higher rating
u_stat, p_value = mannwhitneyu(hot_ratings, not_hot_ratings, alternative='greater')

print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"P-value: {p_value}")

#calculate effect size
N = len(hot_ratings) + len(not_hot_ratings)
z = (u_stat - (len(hot_ratings) * len(not_hot_ratings)) / 2) / np.sqrt(len(hot_ratings) * len(not_hot_ratings) * (len(hot_ratings) + len(not_hot_ratings) + 1) / 12)
r = z / np.sqrt(N)

print(f"Rank-Biserial Correlation (r): {r}")
#relatively large effect size

#boxplot 
#plt.boxplot([hot_ratings, not_hot_ratings], labels=['Hot Ratings', 'Not Hot Ratings'])
#plt.ylabel('Average Rating')
#plt.title('Hot/Not Hot Average Ratings')
#plt.show()

# Visualize figure
boxplots = plt.boxplot(
    [hot_ratings, not_hot_ratings],
    labels=['Hot Ratings', 'Not Hot Ratings'],
    widths=0.6,
    patch_artist=True
)

# Define colors for the boxplots
colors = ['lightblue', 'lightgreen']  # Colors for High Online and Low Online boxplots
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Annotate medians
medians = boxplots['medians']  # Median lines
for i, median in enumerate(medians):
    median_value = median.get_ydata()[0]  # Extract the median value
    plt.text(
        i + 1,  # x-coordinate (boxplot index starts at 1)
        median_value,  # y-coordinate
        f'{median_value:.2f}',  # Format the label
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        color='black',  # Label color
        fontsize=10  # Font size
    )

# Add labels and title
plt.ylabel('Average Rating')
plt.title('Hot/Not Hot Average Ratings')
plt.show()

#%%
#Q7
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

regression_df = df[['average difficulty', 'average rating']].dropna()

X = regression_df[['average difficulty']]  # Predictor
y = regression_df['average rating']       # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N_number)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R^2: {r2}")
print(f"RMSE: {rmse}")

plt.scatter(X_test, y_test, color='blue', label='Actual Ratings')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Linear Regression: Difficulty vs. Rating')
plt.legend()
plt.show()

print(model.coef_)


#%%
#Q8
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#prep data
quant_df_clean = df.dropna()

# Separate predictors (X) and target (y)
X = quant_df_clean.drop(columns=['average rating'])  # Drop the target variable
y = quant_df_clean['average rating']

# Standardize the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add a constant to the predictors for the intercept
X_scaled_with_const = sm.add_constant(X_scaled)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled_with_const, y, test_size=0.2, random_state=N_number)

# Fit multiple linear regression model using statsmodels (OLS)
model = sm.OLS(y_train, X_train).fit()

# Make predictions (for testing purposes)
y_pred_multi = model.predict(X_test)

# Calculate R² and RMSE
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

# Print model summary for p-values and coefficients
print(model.summary())

# Print the R² and RMSE values
print(f"R² (All Factors): {r2_multi:.4f}")
print(f"RMSE (All Factors): {rmse_multi:.4f}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=N_number)

#fit multiple linear regression model
multi_reg_model = LinearRegression()
multi_reg_model.fit(X_train, y_train)

# Make predictions
y_pred_multi = multi_reg_model.predict(X_test)

# Calculate R² and RMSE
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print(f"R² (All Factors): {r2_multi:.4f}")
print(f"RMSE (All Factors): {rmse_multi:.4f}")

#display the coefficients with their column names for clarity
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Beta': model.params[1:],  # Ignore the first constant term as it's not a feature
    'p-value': model.pvalues[1:]
})

# Sort by absolute value of Beta for clarity 
coefficients['Abs Beta'] = coefficients['Beta'].abs()
coefficients = coefficients.sort_values(by='Abs Beta', ascending=False)

print("\nCoefficients and p-values:")
print(coefficients)

#check for colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print(vif_data)

import matplotlib.pyplot as plt

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_multi, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings (Multiple Linear Regression)')
plt.show()



#%%
#Q9
#logistic regression

pepper_distribution = df['received a pepper?'].value_counts()
print(pepper_distribution)


from sklearn.model_selection import train_test_split

# Features and target
X = df[['average rating']]  # Feature(s)
y = df['received a pepper?']  # Target (binary)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=N_number, stratify=y)

from sklearn.linear_model import LogisticRegression

# Model with class weights to address imbalance (optional)
model = LogisticRegression(class_weight='balanced', random_state=N_number)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of "pepper = 1"

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_auc_score
print("AUROC:", roc_auc_score(y_test, y_pred_proba))
auc = roc_auc_score(y_test, y_pred_proba)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AU(RO)C = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print("Precision:", precision)


#%%
#Q10
#logistic regression

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

# Prepare data
X = df.drop(columns=['received a pepper?'])  # All available factors except the target
X = X.dropna()
y = df['received a pepper?']
y = y[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=N_number, stratify=y)

# Model training
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=N_number)
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate metrics
auc = roc_auc_score(y_test, y_pred_proba)
print("AU(RO)C:", auc)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AU(RO)C = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random Classifier", color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%
#Extra credit

#start with original 2 dfs

#giving labels to the columns so its easier to extract
labels = ['major/field', 'university', 'US state']
qual_df_labels = qualitative_df
qual_df_labels.columns = labels

labels2 = ['average rating', 'average difficulty', 'number of ratings', 'received a pepper?', 
          'proportion of students that would take class again', 'number of ratings from online classes',
          'male gender', 'female gender']
quant_df_labels_2 = quantitative_df
quant_df_labels_2.columns = labels2

#merge dfs
merged_df = pd.concat([quantitative_df, qualitative_df], axis=1)

chem_df = merged_df[merged_df['major/field'] == 'Chemistry']
psych_df = merged_df[merged_df['major/field'] == 'Psychology']

#threshold
chem_df_threshold = chem_df[chem_df['number of ratings'] >= 3]
psych_df_threshold = psych_df[psych_df['number of ratings'] >= 3]

#drop nas
chem_ratings = chem_df_threshold['average rating'].dropna()
psych_ratings = psych_df_threshold['average rating'].dropna()

#check normality
#visualize the distribution of ratings for both groups
plt.figure(figsize=(10, 6))
sns.histplot(chem_ratings, kde=True, label='Chemistry', color='blue', stat="density")
sns.histplot(psych_ratings, kde=True, label='Psychology', color='red', stat="density")
plt.legend()
plt.title('Distribution of Ratings for Chemistry and Psychology')
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.show()

#check variance
levene_stat, levene_p_value = stats.levene(chem_ratings, psych_ratings)
print(f"Levene's Test for Equal Variances: Statistic={levene_stat:.4f}, p-value={levene_p_value:.4f}")

#test
stat, p_value = stats.mannwhitneyu(chem_ratings, psych_ratings, alternative='less')
print(f"Mann-Whitney U statistic: {stat}, p-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: Chem ratings are significantly lower than psych ratings")
else:
    print("Fail to reject the null hypothesis: No significant difference between chem and psych ratings.")


#calculate effect size
# Calculate means and standard deviations
mean_chem = np.mean(chem_ratings)
mean_psych = np.mean(psych_ratings)
std_chem = np.std(chem_ratings, ddof=1)  # ddof=1 for sample standard deviation
std_psych = np.std(psych_ratings, ddof=1)

# Calculate Cohen's d
d = (mean_chem - mean_psych) / np.sqrt((std_chem**2 + std_psych**2) / 2)
print(f"Cohen's d: {d}")

#show boxplot
plt.figure(figsize=(8, 6))

# Create the boxplot with specified widths and colors
box = plt.boxplot([chem_ratings, psych_ratings], 
                  labels=['Chemistry', 'Psychology'], 
                  patch_artist=True, 
                  widths=0.6,  # Set the width of the boxplots
                  boxprops=dict(color='black'),  # Box edge colors
                  medianprops=dict(color='orange', linewidth=2),  # Median properties
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black'))

# Change box colors for Chemistry and Psychology
colors = ['lightyellow', 'lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Add median values as text
medians = [round(np.median(chem_ratings), 2), round(np.median(psych_ratings), 2)]
for i, median in enumerate(medians):
    plt.text(i + 1, median, f'{median}', color='black', ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.ylabel('Average Rating')
plt.title('Comparison of Average Ratings: Chemistry vs Psychology')

plt.show()




