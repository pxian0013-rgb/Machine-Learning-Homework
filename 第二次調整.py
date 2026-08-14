import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# =============================================================================
# 1. 讀取資料
# =============================================================================
train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')

# =============================================================================
# 2. 深度特徵工程（注意：train_df 與 test_df 都要處理！）
# =============================================================================
for df in [train_df, test_df]:  # 修正：改成同時處理 train 與 test
  # (A) 補缺值 (用 train 的中位數補，避免數據洩露)
  df['Age'] = df['Age'].fillna(train_df['Age'].median())
  df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

  # (B) 類別轉數值
  df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

  # (C) 衍生新特徵
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
  df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

  # (D) 從姓名抓取稱謂特徵
  df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(
      [
          'Lady',
          'Countess',
          'Capt',
          'Col',
          'Don',
          'Dr',
          'Major',
          'Rev',
          'Sir',
          'Jonkheer',
          'Dona',
      ],
      'Rare',
  )
  df['Title'] = df['Title'].replace('Mlle', 'Miss')
  df['Title'] = df['Title'].replace('Ms', 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
  df['Title'] = df['Title'].map(title_mapping).fillna(0)

# 準備特徵矩陣
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']
X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]  # 新增：考卷的特徵題目

# =============================================================================
# 3. 隨機森林超參數調優
# =============================================================================
# 防過擬合策略：把 max_depth 調淺一點 (例如 3, 4, 5, 6)，在 Kaggle 實測分數會更好喔！
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=1
)
grid_search.fit(X_train, y_train)

print(f'衝刺後最佳交叉驗證準確率: {grid_search.best_score_:.4f}')
print('最佳超參數:', grid_search.best_params_)

# =============================================================================
# 4. 預測考卷並匯出 Kaggle 上傳檔（補上的部分）
# =============================================================================
# (A) 取得調優後的最佳模型
best_model = grid_search.best_estimator_

# (B) 進行預測
print('正在對測試集 (test.csv) 進行預測...')
final_predictions = best_model.predict(X_test)

# (C) 製作 Kaggle 格式的 DataFrame
submission = pd.DataFrame(
    {'PassengerId': test_df['PassengerId'], 'Survived': final_predictions}
)

# (D) 存成 CSV 檔
submission_path = 'C:/Users/陳品賢/Desktop/my_submission.csv'
submission.to_csv(submission_path, index=False)

print('-' * 40)
print(f'✅ Kaggle 答案卡已成功生成於：{submission_path}')
print('現在可以把桌面的 my_submission.csv 拿去 Kaggle 提交囉！')
print('-' * 40)