import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. 讀取資料
try:
  train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
  test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')
except FileNotFoundError:
  train_df = pd.read_csv('C:/titanic/train.csv')
  test_df = pd.read_csv('C:/titanic/test.csv')

# 2. 精簡且乾淨的特徵工程 (避免過度複雜化)
for df in [train_df, test_df]:
  # (A) 年齡補中位數
  df['Age'] = df['Age'].fillna(train_df['Age'].median())
  df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

  # (B) 性別轉數字
  df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

  # (C) 簡化版的稱謂 (只分 Mr, Mrs/Miss, Master, 其他)
  df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(
      ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
      'Rare',
  )
  df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(0)

  # (D) 家庭人數
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 選取關鍵特徵 (不過度堆疊)
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title']
X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]

# 3. 改回隨機森林，並限制樹的深度 (嚴格防止過擬合！)
rf_model = RandomForestClassifier(random_state=42)

# 關鍵調優：縮減樹的深度 (max_depth 設小一點，泛化能力反而超強)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],  # 限制深度在 3~5 之間！
    'min_samples_leaf': [2, 4, 6],  # 規定葉子節點最少人數，防個案記憶
    'min_samples_split': [5, 10],
}

grid_search = GridSearchCV(
    rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=1
)
grid_search.fit(X_train, y_train)

print(f'本地 CV 評分: {grid_search.best_score_:.4f}')
print('最佳超參數:', grid_search.best_params_)

# 4. 輸出預測
best_model = grid_search.best_estimator_
final_predictions = best_model.predict(X_test)

submission = pd.DataFrame(
    {'PassengerId': test_df['PassengerId'], 'Survived': final_predictions}
)
submission.to_csv('C:/Users/陳品賢/Desktop/my_submission.csv', index=False)
print('✅ 答案卡已更新，請重新上傳 Kaggle 測試！')