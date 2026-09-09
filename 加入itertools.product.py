import itertools
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==========================================
# 1. 定位桌面並載入資料
# ==========================================
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
os.chdir(desktop_path)

print("正在讀取桌面上的 train.csv 與 test.csv...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 合併資料集統一做特徵處理
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# ==========================================
# 2. 深度特徵工程 (新增人均票價、票號前綴、團體生存訊號)
# ==========================================
print("進行深度特徵工程與清理...")

# (1) 提取頭銜
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

# (2) 補全年齡與票價 (使用分組中位數)
df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Fare'] = df.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)

# (3) 【新增項目】人均票價 (Fare_Per_Person)
df['Ticket_Count'] = df.groupby('Ticket')['Ticket'].transform('count')
df['Fare_Per_Person'] = df['Fare'] / df['Ticket_Count']

# (4) 【新增項目】票號前綴 (Ticket_Prefix)
df['Ticket_Prefix'] = df['Ticket'].apply(
    lambda x: (
        ''.join(x.split(' ')[:-1])
        .replace('.', '')
        .replace('/', '')
        .upper()
        .strip()
        if len(x.split(' ')) > 1
        else 'NONE'
    )
)

# (5) 【新增項目】團體生存訊號 (Group_Survival_Signal)
df['Is_Woman_Or_Child'] = (
    (df['Sex'] == 'female') | (df['Age'] <= 12)
).astype(int)
df['Group_Survival_Signal'] = 0

groups = df[df['Ticket_Count'] > 1].groupby('Ticket')
for ticket, group in groups:
  train_members = group[group['Survived'].notnull()]
  if len(train_members) > 0:
    wc_members = train_members[train_members['Is_Woman_Or_Child'] == 1]
    if len(wc_members) > 0:
      if wc_members['Survived'].max() == 1:
        df.loc[df['Ticket'] == ticket, 'Group_Survival_Signal'] = 1
      elif wc_members['Survived'].min() == 0:
        df.loc[df['Ticket'] == ticket, 'Group_Survival_Signal'] = -1

# (6) 計算家庭人數與單身標記
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# (7) 船艙與港口補全
df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# (8) 類別變數數值化 (One-Hot Encoding)
df['Sex_Code'] = df['Sex'].map({'female': 0, 'male': 1})
df = pd.get_dummies(
    df, columns=['Title', 'Embarked', 'Ticket_Prefix'], drop_first=True
)

# ==========================================
# 3. 準備模型訓練資料 (定義 X 與 y)
# ==========================================
train_df = df[df['Survived'].notnull()].copy()
test_df = df[df['Survived'].isnull()].copy()

# 選取用於訓練的欄位特徵
features = [
    c
    for c in train_df.columns
    if c not in ['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin']
]

X = train_df[features]
y = train_df['Survived'].astype(int)
X_test = test_df[features]

# ==========================================
# 4. 【新增項目】使用 itertools.product 進行網格搜尋
# ==========================================
print("\n" + "=" * 40)
print("使用 itertools.product 開始搜尋最佳超參數...")
print("=" * 40)

# 定義要搜尋的超參數範圍
param_grid = {
    'n_estimators': [30, 60, 90, 120, 150 ],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
}

# 拆解字典鍵與值的組合
keys, values = zip(*param_grid.items())

best_score = 0.0
best_params = None
cv = StratifiedKFold(n_splits=5, shuffle=True)

# 利用 itertools.product 生成全排列組合
for v in itertools.product(*values):
  current_params = dict(zip(keys, v))

  # 建立 Random Forest 模型（設為 n_jobs=1 避免路徑編碼錯誤）
  model = RandomForestClassifier(**current_params, n_jobs=1)

  # 計算 5 折交叉驗證分數
  scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
  mean_score = scores.mean()

  # 紀錄最高的 CV 分數與組合
  if mean_score > best_score:
    best_score = mean_score
    best_params = current_params

print(f"搜尋完成！最高 5 折交叉驗證分數: {best_score:.4f}")
print(f"最佳參數組合: {best_params}")
print("=" * 40)

# ==========================================
# 5. 使用最佳參數訓練最終模型並生成答案卡
# ==========================================
print("帶入最佳參數訓練最終模型...")
best_model = RandomForestClassifier(**best_params, n_jobs=1)
best_model.fit(X, y)

# 預測考卷
final_predictions = best_model.predict(X_test)

# 輸出 submission.csv
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': final_predictions,
})

submission.to_csv('submission.csv', index=False)

print("完成！已於桌面生成解答檔案 'submission.csv'")
print(f"測試集總人數: {len(submission)} 人")
print(f"預測生存人數: {(submission['Survived'] == 1).sum()} 人")
print(f"預測遇難人數: {(submission['Survived'] == 0).sum()} 人")
print("=" * 40)