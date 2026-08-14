import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')

all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 2. 稱謂與基礎特徵
all_df['Title'] = all_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_df['Title'] = all_df['Title'].replace(
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
all_df['Title'] = all_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_df['Title'] = all_df['Title'].replace('Mme', 'Mrs')

# 標記是否為女性或小孩 (Woman or Child)
all_df['IsWomanOrChild'] = (
    (all_df['Sex'] == 'female') | (all_df['Age'] < 12)
).astype(int)

# 3. 提取姓氏與團體 ID
all_df['Surname'] = all_df['Name'].apply(lambda x: x.split(',')[0].strip())
# 用「姓氏+票價」作為更準確的家族辨識指標
all_df['FamilyID'] = (
    all_df['Surname'] + '_' + all_df['Fare'].astype(str)
)

# 4. 精細計算「女性與小孩」的團體存活率
all_df['WomanChildGroupSurvival'] = (
    0.5  # 預設 0.5 (代表成年男性或單身者，維持預設規則)
)

for fid, group in all_df.groupby('FamilyID'):
  if len(group) > 1:
    # 只針對「婦幼團體」計算連帶存活率
    w_c_group = group[group['IsWomanOrChild'] == 1]
    if len(w_c_group) > 0:
      for idx, row in group.iterrows():
        # 排除自己的其他人存活率
        other_survived = w_c_group.loc[
            w_c_group['PassengerId'] != row['PassengerId'], 'Survived'
        ].dropna()
        if len(other_survived) > 0:
          all_df.loc[idx, 'WomanChildGroupSurvival'] = other_survived.mean()

# 5. 補缺值與轉碼
all_df['Sex'] = all_df['Sex'].map({'male': 1, 'female': 0})
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].median())
all_df['Fare'] = all_df['Fare'].fillna(all_df['Fare'].median())
all_df['Title'] = (
    all_df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(0)
)

# 6. 拆回 Train / Test
train_clean = all_df[all_df['Survived'].notnull()].copy()
test_clean = all_df[all_df['Survived'].isnull()].copy()

features = [
    'Pclass',
    'Sex',
    'Age',
    'Fare',
    'Title',
    'WomanChildGroupSurvival',
]
X_train = train_clean[features]
y_train = train_clean['Survived']
X_test = test_clean[features]

# 7. 訓練模型 (淺層樹)
model = RandomForestClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
model.fit(X_train, y_train)

# 8. 輸出
test_clean['Survived'] = model.predict(X_test).astype(int)
submission = test_clean[['PassengerId', 'Survived']]
submission.to_csv('C:/Users/陳品賢/Desktop/my_submission.csv', index=False)

print('✅ 精細版婦幼團體邏輯已輸出，可以再次上傳 Kaggle 測試！')