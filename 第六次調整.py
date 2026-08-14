import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')

# 合併資料集以便全域計算「家族/團體」
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 2. 抓出姓氏 (Surname) 與 船票 (Ticket)
all_df['Surname'] = all_df['Name'].apply(lambda x: x.split(',')[0].strip())

# 3. 定義「團體」：如果船票相同，或是姓氏相同且家庭人數 > 1
all_df['Group'] = all_df['Ticket']

# 找出人數 > 1 的團體（代表有人同行）
group_counts = all_df['Group'].value_counts()
all_df['GroupSize'] = all_df['Group'].map(group_counts)

# 4. 計算團體內「其他人」的存活情況 (Group Survival Feature)
# 邏輯：看同團體裡除了自己之外，其他人的平均存活率
all_df['Group_Survival'] = 0.5  # 預設中立值 (0.5)

for group, group_df in all_df.groupby('Group'):
  if len(group_df) > 1:  # 如果是團體
    for idx, row in group_df.iterrows():
      smask = group_df['PassengerId'] != row['PassengerId']
      other_survived = group_df.loc[smask, 'Survived'].dropna()
      if len(other_survived) > 0:
        all_df.loc[idx, 'Group_Survival'] = other_survived.mean()

# 5. 基本特徵清理
all_df['Sex'] = all_df['Sex'].map({'male': 1, 'female': 0})
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].median())
all_df['Fare'] = all_df['Fare'].fillna(all_df['Fare'].median())

# 拆回 train / test
train_clean = all_df[all_df['Survived'].notnull()].copy()
test_clean = all_df[all_df['Survived'].isnull()].copy()

# 選取包含「團體存活率」的關鍵特徵
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Group_Survival']
X_train = train_clean[features]
y_train = train_clean['Survived']
X_test = test_clean[features]

# 6. 使用最簡單受限的隨機森林（防過擬合）
rf = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42
)
rf.fit(X_train, y_train)

# 預測並輸出
test_clean['Survived'] = rf.predict(X_test).astype(int)
submission = test_clean[['PassengerId', 'Survived']]
submission.to_csv('C:/Users/陳品賢/Desktop/my_submission.csv', index=False)

print('✅ 已加入「團體連帶存活特徵」，請重新提交 my_submission.csv 試試看！')