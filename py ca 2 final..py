import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.style.use("dark_background")

# =====================================================
# 1️⃣ LOAD DATA (WORKS FOR ANY FILE)
# =====================================================
file_path = "C:/Users/HAI/Downloads/python data set ca2/Exam_Score_Prediction.csv"
df = pd.read_csv(file_path)

print("\n================ DATA LOADED ================\n")
print("Shape:", df.shape)
print(df.head(10))

# Clean column names
df.columns = df.columns.str.strip().str.replace(".", "", regex=False).str.replace(" ", "_", regex=False)

# =====================================================
# 2️⃣ COLUMN IDENTIFICATION
# =====================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

print("\n================ COLUMN DETAILS ================\n")
print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", cat_cols)

# =====================================================
# 3️⃣ BIG OUTPUT SECTION (UNIVERSAL)
# =====================================================

print("\n================ SUMMARY STATISTICS ================\n")
print(df.describe(include="all"))

print("\n================ MISSING VALUES REPORT ================\n")
print(df.isnull().sum())

print("\n================ CORRELATION TABLE ================\n")
if len(numeric_cols) > 1:
    print(df[numeric_cols].corr())
else:
    print("Not enough numeric columns for correlation.")

print("\n================ CATEGORY VALUE COUNTS ================\n")
for col in cat_cols:
    print(f"\nValue Counts for {col}:\n")
    print(df[col].value_counts())

# =====================================================
# 4️⃣ 8-GRAPHS (AUTO SKIPS INVALID ONES)
# =====================================================

# ⭐ GRAPH 1 — Histogram
if len(numeric_cols) > 0:
    plt.figure(figsize=(7,5))
    sns.histplot(df[numeric_cols[0]], bins=30)
    plt.title(f"Graph 1 — Histogram of {numeric_cols[0]}")
    plt.show()

# ⭐ GRAPH 2 — Boxplot
if len(numeric_cols) > 0:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Graph 2 — Boxplot of Numeric Columns")
    plt.xticks(rotation=45)
    plt.show()

# ⭐ GRAPH 3 — Heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Graph 3 — Correlation Heatmap")
    plt.show()

# ⭐ GRAPH 4 — Scatter Plot
if len(numeric_cols) > 1:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[-1]])
    plt.title(f"Graph 4 — {numeric_cols[0]} vs {numeric_cols[-1]}")
    plt.show()

# ⭐ GRAPH 5 — Countplot
if len(cat_cols) > 0:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=cat_cols[0])
    plt.title(f"Graph 5 — Countplot of {cat_cols[0]}")
    plt.xticks(rotation=45)
    plt.show()

# ⭐ GRAPH 6 — Pairplot
if len(numeric_cols) >= 2:
    sns.pairplot(df[numeric_cols])
    plt.suptitle("Graph 6 — Pairplot", y=1.02)
    plt.show()

# ⭐ GRAPH 7 — Bar Chart
if len(cat_cols) > 0:
    plt.figure(figsize=(10,5))
    df.groupby(cat_cols[0])[numeric_cols[-1]].mean().sort_values().plot(kind="bar")
    plt.title(f"Graph 7 — Mean {numeric_cols[-1]} by {cat_cols[0]}")
    plt.show()

# ⭐ GRAPH 8 — Regression Plot
if len(numeric_cols) > 1:
    plt.figure(figsize=(7,5))
    sns.regplot(x=df[numeric_cols[0]], y=df[numeric_cols[-1]])
    plt.title(f"Graph 8 — Regression: {numeric_cols[-1]} vs {numeric_cols[0]}")
    plt.show()

# =====================================================
# 5️⃣ MACHINE LEARNING (AUTOMATIC)
# =====================================================
print("\n================ MACHINE LEARNING MODEL ================\n")

# Auto-select target = last numeric column
target = numeric_cols[-1]
features = [c for c in numeric_cols if c != target]

print("Target Variable:", target)
print("Features:", features)

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =====================================================
# 6️⃣ ERROR METRICS (BIG OUTPUT)
# =====================================================
print("\n================ MODEL PERFORMANCE ================\n")
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
    "Value": [
        mean_absolute_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        r2_score(y_test, y_pred)
    ]
})
print(metrics_df)

# =====================================================
# 7️⃣ FEATURE IMPORTANCE
# =====================================================
print("\n================ FEATURE IMPORTANCE ================\n")
importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)
print(importance)

# =====================================================
# 8️⃣ ACTUAL VS PREDICTED GRAPH
# =====================================================
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='cyan')
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b, color='yellow', linestyle='--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.show()


