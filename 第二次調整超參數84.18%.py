import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. 讀取資料
train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')

# 2. 深度特徵工程
for df in [train_df]:
  # (A) 補缺值
  df['Age'] = df['Age'].fillna(df['Age'].median())
  df['Fare'] = df['Fare'].fillna(df['Fare'].median())

  # (B) 類別轉數值
  df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

  # (C) 衍生新特徵
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
  df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

  # (D) 從姓名抓取稱謂特徵
  df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(
      ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
      'Rare',
  )
  df['Title'] = df['Title'].replace('Mlle', 'Miss')
  df['Title'] = df['Title'].replace('Ms', 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
  df['Title'] = df['Title'].map(title_mapping).fillna(0)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']
X_train = train_df[features]
y_train = train_df['Survived']

# 3. 隨機森林超參數調優
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
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