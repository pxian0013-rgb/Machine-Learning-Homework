import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# =============================================================================
# 1. 讀取資料
# =============================================================================
train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')

# =============================================================================
# 2. 進階特徵工程（突破 0.80 的關鍵）
# =============================================================================
for df in [train_df, test_df]:
  # (A) 補缺值
  df['Age'] = df['Age'].fillna(train_df['Age'].median())
  df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

  # (B) 性別轉數字
  df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

  # (C) 稱謂提取
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
  df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  df['Title'] = (
      df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(0)
  )

  # (D) 家庭人數與孤身一人
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
  df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

  # (E) 【新特徵】票價分級（離散化，避免極端數值干擾）
  df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)

  # (F) 【新特徵】年齡 x 艙等（交互作用）
  df['Age_Class'] = df['Age'] * df['Pclass']

# 選擇特徵
features = [
    'Pclass',
    'Sex',
    'Age',
    'FareBin',
    'FamilySize',
    'IsAlone',
    'Title',
    'Age_Class',
]
X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]

# =============================================================================
# 3. 梯度提升模型 (Gradient Boosting) + 學習率 (learning_rate) 調優
# =============================================================================
print('🚀 開始訓練梯度提升模型 (GradientBoostingClassifier)...')

# 宣告模型
gb_model = GradientBoostingClassifier(random_state=42)

# 定義超參數搜尋範圍（包含 learning_rate）
param_grid = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],  # 學習率：每次疊代的步幅
    'n_estimators': [100, 150, 200],  # 樹的棵數
    'max_depth': [3, 4],  # 樹深控制在 3~4，嚴防過擬合
    'subsample': [0.8, 1.0],  # 每棵樹隨機抽取的資料比例
    'min_samples_split': [4, 6],
}

# 執行網格搜尋
grid_search = GridSearchCV(
    gb_model, param_grid, cv=5, scoring='accuracy', n_jobs=1
)
grid_search.fit(X_train, y_train)

print('-' * 40)
print(f'本地 CV 最佳準確率: {grid_search.best_score_:.4f}')
print('最佳超參數組合:', grid_search.best_params_)
print('-' * 40)

# =============================================================================
# 4. 預測並輸出 Kaggle 答案卡
# =============================================================================
best_model = grid_search.best_estimator_
final_predictions = best_model.predict(X_test)

submission = pd.DataFrame(
    {'PassengerId': test_df['PassengerId'], 'Survived': final_predictions}
)

submission_path = 'C:/Users/陳品賢/Desktop/my_submission.csv'
submission.to_csv(submission_path, index=False)

print(f'✅ 新版答案卡已成功生成於：{submission_path}')
print('這次引入了 learning_rate 與新特徵，去提交看看吧！')