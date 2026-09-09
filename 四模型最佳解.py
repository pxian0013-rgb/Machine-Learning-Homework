import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ==========================================
# 1. 定位桌面並載入資料
# ==========================================
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
os.chdir(desktop_path)

print("正在讀取桌面上的 train.csv 與 test.csv...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df = pd.concat([train, test], sort=False).reset_index(drop=True)

# ==========================================
# 2. 深度特徵工程
# ==========================================
print("進行深度特徵提取與組合...")

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

# (2) 填補缺漏值
df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Fare'] = df.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# (3) 家庭規模與人均票價
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Ticket_Count'] = df.groupby('Ticket')['Ticket'].transform('count')
df['Fare_Per_Person'] = df['Fare'] / df['Ticket_Count']

# (4) 年齡細分與交叉特徵
df['Is_Child'] = (df['Age'] <= 12).astype(int)
df['Is_Elderly'] = (df['Age'] >= 60).astype(int)
df['Age_Class'] = df['Age'] * df['Pclass']

# (5) 船艙甲板與票號前綴
df['Deck'] = df['Cabin'].apply(
    lambda x: str(x)[0] if pd.notna(x) else 'Unknown'
)
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

# (6) 團體生存訊號 (Group Survival Signal)
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

# (7) 類別轉碼
df['Sex_Code'] = df['Sex'].map({'female': 0, 'male': 1})
df = pd.get_dummies(
    df,
    columns=['Title', 'Embarked', 'Deck', 'Pclass', 'Ticket_Prefix'],
    drop_first=True,
)

# ==========================================
# 3. 準備模型資料集
# ==========================================
train_df = df[df['Survived'].notnull()].copy()
test_df = df[df['Survived'].isnull()].copy()

drop_cols = ['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin']
features = [c for c in train_df.columns if c not in drop_cols]

X = train_df[features].values
y = train_df['Survived'].astype(int).values
X_test = test_df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. 帶入 GridSearchCV 搜尋出的黃金超參數
# ==========================================
print("帶入優化後的最佳超參數建立集成模型...")

m1_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=1,
)

m2_et = ExtraTreesClassifier(
    n_estimators=150, max_depth=8, min_samples_split=5
)

m3_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=5,
    subsample=1.0,
    eval_metric='logloss',
    
)

m4_lgb = lgb.LGBMClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=15,
    verbosity=-1,
)

# 依據搜尋出的 CV 分數 (XGB: 0.8777, LGB: 0.8709) 給予更高投票權重
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', m1_rf),
        ('et', m2_et),
        ('xgb', m3_xgb),
        ('lgb', m4_lgb),
    ],
    voting='soft',
    weights=[1, 1, 3, 2],
)

# ==========================================
# 5. 5折交叉驗證與生成最終預測
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(
    ensemble_model, X_scaled, y, cv=cv, scoring='accuracy'
)

print("=" * 40)
print(f"集成模型 5 折交叉驗證平均分數: {scores.mean():.4f}")
print("=" * 40)

# 訓練模型並取得測試集機率
ensemble_model.fit(X_scaled, y)
probabilities = ensemble_model.predict_proba(X_test_scaled)[:, 1]

# 套用驗證最佳的 0.55 ~ 0.60 門檻 (此處預設 0.55)
THRESHOLD = 0.7
final_predictions = (probabilities >= THRESHOLD).astype(int)

# 輸出 submission.csv
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': final_predictions,
})

submission.to_csv('submission.csv', index=False)

print("完成！已更新最終解答檔案 'submission.csv'")
print(f"預測生存人數: {(submission['Survived'] == 1).sum()} 人")
print(f"預測遇難人數: {(submission['Survived'] == 0).sum()} 人")
print("=" * 40)