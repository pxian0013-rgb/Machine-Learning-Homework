import os

os.system("pip install scikit-learn")

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier  # 引入梯度提升模型
from sklearn.model_selection import GridSearchCV  # 引入網格搜尋工具

print("鐵達尼號進階實戰：開始讀取資料...")

# =============================================================================
# 1. 讀取資料
# =============================================================================
try:
  train_df = pd.read_csv("C:/Users/陳品賢/Desktop/train.csv")
  test_df = pd.read_csv("C:/Users/陳品賢/Desktop/test.csv")
except FileNotFoundError:
  train_df = pd.read_csv("C:/titanic/train.csv")
  test_df = pd.read_csv("C:/titanic/test.csv")

# =============================================================================
# 2. 特徵工程（建立衍生特徵，提升上限）
# =============================================================================
for df in [train_df, test_df]:
  # (A) 補缺值
  df["Age"] = df["Age"].fillna(df["Age"].median())
  df["Fare"] = df["Fare"].fillna(df["Fare"].median())

  # (B) 性別轉數字
  df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

  # (C) 家庭人數與獨居狀態
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

  # (D) 稱謂提取
  df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
  df["Title"] = df["Title"].replace(
      [
          "Lady",
          "Countess",
          "Capt",
          "Col",
          "Don",
          "Dr",
          "Major",
          "Rev",
          "Sir",
          "Jonkheer",
          "Dona",
      ],
      "Rare",
  )
  df["Title"] = df["Title"].replace("Mlle", "Miss")
  df["Title"] = df["Title"].replace("Ms", "Miss")
  df["Title"] = df["Title"].replace("Mme", "Mrs")
  title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
  df["Title"] = df["Title"].map(title_mapping).fillna(0)

# 選取特徵
features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Title"]
X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]

# =============================================================================
# 3. 梯度提升模型 + 超參數搜尋（加在這裡！）
# =============================================================================
print("🔍 開始進行 HistGradientBoosting 模型的超參數搜尋...")

# (1) 宣告梯度提升模型（支援 learning_rate 超參數）
hgb_model = HistGradientBoostingClassifier(random_state=42)

# (2) 加入學習率與樹深度的搜尋範圍
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],  # 學習率：控制步幅大小
    "max_iter": [100, 200],  # 相當於 n_estimators（疊代次數/樹的數量）
    "max_depth": [3, 5, 7],  # 樹的最大深度
    "l2_regularization": [0.0, 0.1, 1.0],  # 正則化強度，防止過擬合
}

# (3) 網格搜尋
grid_search = GridSearchCV(
    hgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=1
)
grid_search.fit(X_train, y_train)

print("-" * 40)
print(f"梯度提升模型最佳準確率: {grid_search.best_score_:.4f}")
print("最佳超參數:", grid_search.best_params_)
print("-" * 40)

# 取得搜尋出的最佳模型
best_model = grid_search.best_estimator_

# =============================================================================
# 4. 正式預測並輸出答案卡
# =============================================================================
print("正在預測 test.csv 並輸出答案卡...")
final_predictions = best_model.predict(X_test)

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": final_predictions}
)

submission_path = "C:/Users/陳品賢/Desktop/my_submission.csv"
submission.to_csv(submission_path, index=False)

print(f"✅ 答案卡已成功生成於：{submission_path}")