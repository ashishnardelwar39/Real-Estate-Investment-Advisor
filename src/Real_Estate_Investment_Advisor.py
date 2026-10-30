#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[6]:





# In[9]:


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# In[11]:




# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.head(10)


# In[12]:


df.info


# In[13]:


df.describe


# In[14]:


df.dtypes


# In[15]:


#2. features.py — shared feature engineering
df["Price_per_SqFt"]=df["Price_in_Lakhs"]*1e5/df["Size_in_SqFt"]


# In[16]:


print(df["Price_per_SqFt"])


# In[17]:


df["Amenities_Count"]=df["Amenities"].fillna("").apply(lambda s:len([a for a in s.split(",")if a.strip()]))


# In[18]:


print(df["Amenities_Count"])


# In[19]:


df["Infra_Score"] = (
    df["Nearby_Schools"] / 10
    + df["Nearby_Hospitals"] / 10

) / 2 * 10


# In[20]:


print(df["Infra_Score"])


# In[21]:


df["PPSF_vs_City_Median"] = (
    df["Price_per_SqFt"] /
    df.groupby("City")["Price_per_SqFt"].transform("median")
)


# In[22]:


print(["PPSF_vs_City_Median"])


# In[23]:


df[["City", "Price_per_SqFt", "PPSF_vs_City_Median"]].head()


# In[24]:


#3. Step 1 — Data preprocessing (01_preprocess.py)

df.drop_duplicates()



# In[25]:


#Target 1 — Good_Investment (classification). A multi-factor score of five yes/no rules, good if score ≥ 3:

# Rule 1: Property price per sqft <= median price per sqft of its city
rule1 = (
    df["Price_per_SqFt"]
    <= df.groupby("City")["Price_per_SqFt"].transform("median")
)

# Rule 2: Property price <= median property price of its city
rule2 = (
    df["Price_in_Lakhs"]
    <= df.groupby("City")["Price_in_Lakhs"].transform("median")
)

# Rule 3: BHK >= 3
rule3 = df["BHK"] >= 3

# Rule 4: Ready to move
rule4 = df["Availability_Status"].astype(str).str.strip().str.lower()== "ready_to_move"

# Rule 5: Infrastructure score >= overall median
rule5 = df["Infra_Score"] >= df["Infra_Score"].median()

# Calculate total score (each True = 1, False = 0)
score = (
    rule1.astype(int)
    + rule2.astype(int)
    + rule3.astype(int)
    + rule4.astype(int)
    + rule5.astype(int)
)

# Target: 1 = Good investment, 0 = Not a good investment
df["Good_Investment"] = (score >= 3).astype(int)

# Check results
print(df["Good_Investment"].value_counts())


# In[26]:


print(df["Good_Investment"])


# In[27]:


print(df.columns.tolist())


# In[28]:


df["City_Median_PPSF"] = (
    df.groupby("City")["Price_per_SqFt"].transform("median")
)

df["score"] = (
    df["Price_per_SqFt"] <= df["City_Median_PPSF"]
)


# In[29]:


print(df["City_Median_PPSF"])


# In[30]:


print(df["score"])


# In[31]:


df["Growth_Rate"] = 0.05

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5
)


# In[32]:


#Target 2 — Future_Price_5Y (regression): 

df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5


# In[33]:


print(df["Future_Price_5Y"])


# In[34]:


city_growth = {
    "Mumbai": 0.08,
    "Pune": 0.07,
    "Nagpur": 0.06
}

df["Growth_Rate"] = df["City"].map(city_growth).fillna(0.05)

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5
)


# In[35]:


print(city_growth)

print(df["Growth_Rate"])

print(df["Future_Price_5Y"])


# In[36]:


#EDA
#4. Step 2 — EDA (02_eda.py)

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["Price_in_Lakhs"], kde=True)

plt.title("Distribution of Property Prices")
plt.xlabel("Price in Lakhs")
plt.ylabel("Frequency")
plt.show()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="Property_Type", y="Price_per_SqFt", data=df)

plt.title("Price per SqFt by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Price per SqFt")
plt.xticks(rotation=45)
plt.show()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation
corr = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])

# Create scatter plot
sns.scatterplot(x="Size_in_SqFt", y="Price_in_Lakhs", data=df)

plt.title(f"Property Size vs Price (Correlation = {corr:.2f})")
plt.xlabel("Size (SqFt)")
plt.ylabel("Price (Lakhs)")
plt.show()


# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(y=df["Price_in_Lakhs"])

plt.title("Box Plot of Property Prices")
plt.ylabel("Price (Lakhs)")
plt.show()


# In[40]:


import matplotlib.pyplot as plt

# Calculate average price per square foot by state
state_price = df.groupby("State")["Price_per_SqFt"].mean()

# Create bar chart
state_price.plot(kind="bar", figsize=(10, 5))

plt.title("Average Price per SqFt by State")
plt.xlabel("State")
plt.ylabel("Average Price per SqFt")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[41]:


# Calculate median property age by city and locality
median_age = df.groupby(
    ["City", "Locality"]
)["Age_of_Property"].median()

# Print the result
print(median_age)


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt

# Create cross-tabulation
city_bhk = pd.crosstab(df["City"], df["BHK"])

# Create stacked bar chart
city_bhk.plot(kind="bar", stacked=True, figsize=(10, 6))

plt.title("Distribution of BHK Types by City")
plt.xlabel("City")
plt.ylabel("Number of Properties")
plt.xticks(rotation=45)
plt.legend(title="BHK")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[43]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Find the 5 most expensive localities
top5 = (
    df.groupby(["City", "Locality"])["Price_per_SqFt"]
    .mean()
    .nlargest(5)
    .reset_index()
)

print("Top 5 Most Expensive Localities:")
print(top5)

# 2. Filter data for these localities
top5_df = df.merge(
    top5[["City", "Locality"]],
    on=["City", "Locality"],
    how="inner"
)

# 3. Plot price per SqFt by build year
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=top5_df,
    x="Year_Built",
    y="Price_per_SqFt",
    hue="Locality",
    style="City",
    estimator="mean",
    errorbar=None
)

plt.title("Price per SqFt by Build Year: Top 5 Expensive Localities")
plt.xlabel("Year Built")
plt.ylabel("Average Price per SqFt")
plt.legend(title="Locality / City", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical columns
num = df.select_dtypes(include="number").columns

# Calculate correlation matrix
corr_matrix = df[num].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt

# Average price per SqFt by nearby schools
print("Average Price by Nearby Schools:")
print(df.groupby("Nearby_Schools")["Price_per_SqFt"].mean())

# Average price per SqFt by nearby hospitals
print("\nAverage Price by Nearby Hospitals:")
print(df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean())

# Average price per SqFt by furnishing status
print("\nAverage Price by Furnished Status:")
print(df.groupby("Furnished_Status")["Price_per_SqFt"].mean())

# Average price per SqFt by facing direction
print("\nAverage Price by Facing:")
print(df.groupby("Facing")["Price_per_SqFt"].mean())


# In[46]:


# Count properties by owner type
print("Owner Type Counts:")
print(df["Owner_Type"].value_counts())

# Count properties by availability
print("\nAvailability Counts:")
print(df["Availability_Status"].value_counts())


# In[47]:


print("\nAvailability Counts:")
print(df["Availability_Status"].value_counts())


# In[48]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Average price by parking space
print(df.groupby("Parking_Space")["Price_in_Lakhs"].mean())

# Parking space vs investment
print(pd.crosstab(df["Parking_Space"], df["Good_Investment"]))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=axes[0])
axes[0].set_title("Parking Space vs Average Price")

sns.countplot(data=df, x="Parking_Space", hue="Good_Investment", ax=axes[1])
axes[1].set_title("Parking Space vs Good Investment")

plt.tight_layout()
plt.show()


# In[49]:


# 1. City with highest average price per SqFt
city_price = df.groupby("City")["Price_per_SqFt"].mean()
print(f"{city_price.idxmax()} has the highest average price per sq ft "
      f"({city_price.max():.2f}).")

# 2. Parking space vs average price
parking_price = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
print(f"Properties with {parking_price.idxmax()} parking status "
      f"have the highest average price.")

# 3. Amenities count vs average price
amenity_price = df.groupby("Amenities_Count")["Price_in_Lakhs"].mean()
print(f"Properties with {amenity_price.idxmax()} amenities "
      f"have the highest average price.")

# 4. Transport accessibility vs average price
transport_price = df.groupby("Public_Transport_Accessibility")[
    "Price_in_Lakhs"
].mean()
print(f"Transport accessibility level {transport_price.idxmax()} "
      f"has the highest average property price.")

# 5. Good investment percentage by transport accessibility
investment_rate = df.groupby(
    "Public_Transport_Accessibility"
)["Good_Investment"].apply(
    lambda x: (x.astype(str).str.strip().str.lower()
               .isin(["true", "yes", "1"]).mean()) * 100
)

print("\nGood Investment Rate by Transport Accessibility:")
print(investment_rate.round(2))

# 6. Highest investment rate
best_access = investment_rate.idxmax()
print(f"Transport accessibility level {best_access} has the highest "
      f"good-investment rate ({investment_rate.max():.2f}%).")


# In[50]:


# Numerical columns
NUM_COLS = [
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Price_per_SqFt"
]

# Categorical columns
CAT_COLS = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "Furnished_Status",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing"
]


# In[51]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# In[52]:


#5. Step 3 — Model training & MLflow (03_train.py)

#A. One Pipeline = preprocessing + model

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


prep = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
])

pipe = Pipeline([
    ("prep", prep),
    ("model", model)
])



# In[53]:


# Input features
X= df.drop(columns=["Good_Investment"])

# Target variable
y = df["Good_Investment"]


# In[54]:


X


# In[55]:


y


# In[56]:


#2. Step B — Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[57]:


X_train


# In[58]:


X_test


# In[59]:


y_train


# In[60]:


y_test


# In[61]:


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# In[62]:


get_ipython().run_line_magic('pip', 'install pandas scikit-learn xgboost mlflow joblib')


# In[63]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:


get_ipython().run_line_magic('pip', 'install pandas')


# In[ ]:


import mlflow
print(mlflow.__version__)


# In[ ]:


import sys

print(sys.executable)
print(sys.version)


# In[ ]:


import sys
import platform

print("Python path:", sys.executable)
print("Python version:", sys.version)
print("Architecture:", platform.architecture())


# In[ ]:


try:
    import xgboost as xgb
    print("XGBoost version:", xgb.__version__)
except Exception as e:
    print(type(e).__name__)
    print(e)


# In[ ]:


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

# Classification models
class_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

# Regression models
reg_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}


# In[ ]:


df["City_Median_PPSF"] = (
    df.groupby("City")["Price_per_SqFt"]
      .transform("median")
)

df["Good_Investment"] = (
    df["Price_per_SqFt"] <= df["City_Median_PPSF"]
).map({True: "Good", False: "Bad"})

print(df["Good_Investment"].value_counts())


# In[ ]:


X = df.drop(columns=[
    "Good_Investment",
    "City_Median_PPSF"
])

y = df["Good_Investment"]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[ ]:


# Remove rows without target values
df = df.dropna(subset=["Price_in_Lakhs"])

# Classification target must already be defined
if "Good_Investment" in df.columns:
    df = df.dropna(subset=["Good_Investment"])
else:
    raise ValueError(
        "Create a Good_Investment target before classification."
    )


# In[ ]:


#Step 3: Define input features
from sklearn.model_selection import train_test_split

# Columns that should not be used as input features
exclude_cols = [
    "ID",
    "Good_Investment",
    "Price_in_Lakhs"
]

X = df.drop(columns=exclude_cols, errors="ignore")
y_class = df["Good_Investment"]
y_price = df["Price_in_Lakhs"]

# Identify numerical and categorical features
NUM_COLS = X.select_dtypes(include="number").columns.tolist()
CAT_COLS = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("Numerical columns:", NUM_COLS)
print("Categorical columns:", CAT_COLS)


# In[ ]:


#Step 4: Create the preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", numeric_prep, NUM_COLS),
    ("cat", categorical_prep, CAT_COLS)
])


# In[ ]:


#Step 5: Train all six models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)
import numpy as np

# Separate train/test splits for each task
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42,
    stratify=y_class
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# Classification models
class_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoostClassifier": XGBClassifier(
        n_estimators=100, max_depth=6,
        random_state=42, eval_metric="logloss"
    )
}

# Regression models
reg_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "XGBoostRegressor": XGBRegressor(
        n_estimators=100, max_depth=6,
        random_state=42, objective="reg:squarederror"
    )
}

results = []
best_classifier = None
best_regressor = None
best_f1 = -1
best_r2 = -float("inf")

# Train and evaluate classifiers
for name, model in class_models.items():
    pipe = Pipeline([
        ("prep", prep),
        ("model", model)
    ])

    pipe.fit(X_train_c, y_train_c)
    predictions = pipe.predict(X_test_c)

    accuracy = accuracy_score(y_test_c, predictions)
    f1 = f1_score(
        y_test_c, predictions, average="weighted"
    )

    results.append({
        "Task": "Classification",
        "Model": name,
        "Accuracy": accuracy,
        "F1": f1
    })

    print(name, "Accuracy:", accuracy, "F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        best_classifier = pipe

# Train and evaluate regressors
for name, model in reg_models.items():
    pipe = Pipeline([
        ("prep", prep),
        ("model", model)
    ])

    pipe.fit(X_train_r, y_train_r)
    predictions = pipe.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_r, predictions))
    r2 = r2_score(y_test_r, predictions)

    results.append({
        "Task": "Regression",
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(name, "MAE:", mae, "RMSE:", rmse, "R2:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_regressor = pipe

results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:


#Step 6: Save the best models

best_classifier = pipe

print("Classifier assigned successfully")

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_classifier,
    "models/good_investment_classifier.pkl"
)

joblib.dump(
    best_regressor,
    "models/future_price_regressor.pkl"
)

print("Best models saved successfully.")


# In[ ]:


#Step 7: Compare the results
print(
    results_df.sort_values(
        by="F1",
        ascending=False,
        na_position="last"
    )
)


# In[ ]:


#Track everything with MLflow
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Real_Estate_Investment_Advisor")

with mlflow.start_run(run_name="clf_RandomForest"):
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(pipe, "model")


# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score, f1_score


# In[ ]:





# In[ ]:


# Train the model
pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


num_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", num_prep, NUM_COLS),
    ("cat", cat_prep, CAT_COLS)
])


# In[ ]:


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

pipe = Pipeline([
    ("prep", prep),
    ("model", model)
])


# In[ ]:


#Track everything with MLflow
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Real_Estate_Investment_Advisor")

with mlflow.start_run(run_name="clf_RandomForest"):
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(pipe, "model")


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score


# In[ ]:


# Train the model
pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[ ]:


#This error occurs because City_Median_PPSF is included in your NUM_COLS or CAT_COLS, but it is missing from the DataFrame (X_train) used by your pipeline.

#In your Real Estate Investment Advisor project, you likely removed this column from X but still included it in your preprocessing lists.



# Keep only columns that exist in X_train
NUM_COLS = [
    col for col in NUM_COLS
    if col in X_train.columns
]

CAT_COLS = [
    col for col in CAT_COLS
    if col in X_train.columns
]

print("Numerical columns:", NUM_COLS)
print("Categorical columns:", CAT_COLS)


# In[ ]:


print("Missing numerical columns:",
      set(NUM_COLS) - set(X_train.columns))

print("Missing categorical columns:",
      set(CAT_COLS) - set(X_train.columns))


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), NUM_COLS),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_COLS)
])

pipe = Pipeline([
    ("prep", prep),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

pipe.fit(X_train, y_train)

print("Model trained successfully!")


# In[ ]:


print("Missing numerical columns:",
      set(NUM_COLS) - set(X_train.columns))


# In[ ]:


import joblib

joblib.dump(best_classifier, "models/good_investment_classifier.pkl")
joblib.dump(best_regressor, "models/future_price_regressor.pkl")


# In[ ]:





# In[ ]:





# In[ ]:


import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay
)


# In[ ]:


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Real_Estate_Investment_Advisor")

# Predict on test data
y_pred = best_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Create confusion matrix image
os.makedirs("artifacts", exist_ok=True)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png")
plt.close()

# Log experiment
with mlflow.start_run(run_name="GoodInvestmentClassifier"):
    mlflow.log_param("model_type", "Best Classification Model")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.log_artifact("artifacts/confusion_matrix.png")

    mlflow.sklearn.log_model(
        best_classifier,
        artifact_path="model",
        registered_model_name="GoodInvestmentClassifier"
    )

# Save model locally
os.makedirs("models", exist_ok=True)
joblib.dump(
    best_classifier,
    "models/good_investment_classifier.pkl"
)

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[ ]:


# Predict future prices
y_pred_reg = best_regressor.predict(X_test_reg)

# Calculate regression metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = mean_squared_error(
    y_test_reg, y_pred_reg
) ** 0.5
r2 = r2_score(y_test_reg, y_pred_reg)

# Track regression experiment
with mlflow.start_run(run_name="FuturePriceRegressor"):
    mlflow.log_param("model_type", "Best Regression Model")

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(
        best_regressor,
        artifact_path="model",
        registered_model_name="FuturePriceRegressor"
    )

# Save the model locally
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_regressor,
    "models/future_price_regressor.pkl"
)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# In[ ]:


import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Target column
target = "Good_Investment"

# Remove rows where target is missing
df = df.dropna(subset=[target]).copy()

# Select features and target
X = df.drop(columns=[target])
y = df[target]

# Remove ID if it exists
X = X.drop(columns=["ID"], errors="ignore")

# Identify column types
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# Prepare preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train classifier
best_classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

best_classifier.fit(X_train, y_train)

print("Classifier trained successfully!")


# In[ ]:


import os
import joblib

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Save trained models
joblib.dump(
    best_classifier,
    "models/good_investment_classifier.pkl"
)

joblib.dump(
    best_regressor,
    "models/future_price_regressor.pkl"
)

print("Both Models saved successfully!")


# In[1]:


import os

print(os.listdir("models"))


# In[2]:


import os

print(os.getcwd())
print(os.listdir("models"))


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("Good Investment - Confusion Matrix")
plt.savefig("artifacts/confusion_matrix.png")
plt.show()


# In[ ]:


import mlflow

# Configure MLflow to use SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select an experiment
mlflow.set_experiment("Real_Estate_Investment_Advisor")

print("MLflow tracking URI:", mlflow.get_tracking_uri())

