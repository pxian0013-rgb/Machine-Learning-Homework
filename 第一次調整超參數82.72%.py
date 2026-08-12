import os

os.system("pip install scikit-learn")

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

print("鐵達尼號正式實戰：開始讀取資料...")

# =============================================================================
# 1. 讀取資料（建議把檔案放到全英文路徑，例如 C:/titanic/train.csv）
# =============================================================================
try:
  # 如果你已建立 C:/titanic 資料夾並把 csv 放在裡面：
  train_df = pd.read_csv("C:/titanic/train.csv")
  test_df = pd.read_csv("C:/titanic/test.csv")
except FileNotFoundError:
  # 備用方案：若還沒搬移，先嘗試讀取桌面
  train_df = pd.read_csv("C:/Users/陳品賢/Desktop/train.csv")
  test_df = pd.read_csv("C:/Users/陳品賢/Desktop/test.csv")

# =============================================================================
# 2. 特徵工程
# =============================================================================
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["Sex"] = train_df["Sex"].map({"male": 1, "female": 0})

test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
test_df["Sex"] = test_df["Sex"].map({"male": 1, "female": 0})
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]

# =============================================================================
# 3. 超參數調整與模型訓練
# =============================================================================
print("🔍 開始進行超參數搜尋與交叉驗證...")

param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

base_model = DecisionTreeClassifier(random_state=42)

# 注意：這裡將 n_jobs 改為 1，就不會再報 UnicodeEncodeError 錯了！
grid_search = GridSearchCV(
    estimator=base_model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=1
)

grid_search.fit(X_train, y_train)

print("-" * 40)
print(f"最佳準確率 (CV Score): {grid_search.best_score_:.4f}")
print("最佳超參數組合：", grid_search.best_params_)
print("-" * 40)

best_model = grid_search.best_estimator_

# =============================================================================
# 4. 預測與輸出答案卡
# =============================================================================
print("正在預測 test.csv...")
final_predictions = best_model.predict(X_test)

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": final_predictions}
)

# 輸出路徑（同樣建議改至全英文資料夾）
try:
  submission_path = "C:/titanic/my_submission.csv"
  submission.to_csv(submission_path, index=False)
except Exception:
  submission_path = "C:/Users/陳品賢/Desktop/my_submission.csv"
  submission.to_csv(submission_path, index=False)

print(f"答案卡已成功生成於：{submission_path}")