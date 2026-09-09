import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
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
# 2. 深度特徵工程 (Deep Feature Engineering)
# ==========================================
print("進行深度特徵提取與組合...")

# (1) 提取頭銜與姓氏
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

# (2) 填補缺漏值 (Age, Fare, Embarked)
df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Fare'] = df.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# (3) 建立家庭規模與人均票價 (Fare per Person)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 計算票號重複次數（同張票可能多人使用）
df['Ticket_Count'] = df.groupby('Ticket')['Ticket'].transform('count')
# 算出真的人均票價（解決家族合票導致 Fare 虛高問題）
df['Fare_Per_Person'] = df['Fare'] / df['Ticket_Count']

# (4) 年齡細分 (Age Grouping)
df['Is_Child'] = (df['Age'] <= 12).astype(int)
df['Is_Elderly'] = (df['Age'] >= 60).astype(int)

# (5) 交叉互動特徵 (Feature Interaction)
# 年齡與艙等的乘積 (Age * Pclass)
df['Age_Class'] = df['Age'] * df['Pclass']

# (6) 船艙與甲板特徵 (Deck)
df['Deck'] = df['Cabin'].apply(
    lambda x: str(x)[0] if pd.notna(x) else 'Unknown'
)

# (7) 票號前綴處理 (Ticket Prefix)
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

# (8) 團體生存訊號 (Group Survival Signal)
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

# (9) 類別轉碼
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
# 4. 建立現代競賽強效集成模型 (XGBoost + LightGBM + RF + ExtraTrees)
# ==========================================
print("建立多模型集成 (Ensemble)...")

m1_rf = RandomForestClassifier(
    n_estimators=100, max_depth=6, min_samples_split=10
    )
m2_et = ExtraTreesClassifier(
    n_estimators=100, max_depth=6, min_samples_split=10
)
m3_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    eval_metric='logloss',
)
m4_lgb = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.03, max_depth=4,verbosity=-1
)

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', m1_rf),
        ('et', m2_et),
        ('xgb', m3_xgb),
        ('lgb', m4_lgb),
    ],
    voting='soft',
    weights=[2, 1, 2, 2],
)

# ==========================================
# 5. 評估與預測 (門檻優化)
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(
    ensemble_model, X_scaled, y, cv=cv, scoring='accuracy'
)

print("=" * 40)
print(f"交叉驗證平均準確率: {scores.mean():.4f}")
print("=" * 40)

# 訓練並取得測試集生存概率
ensemble_model.fit(X_scaled, y)
probabilities = ensemble_model.predict_proba(X_test_scaled)[:, 1]

THRESHOLD = 0.7
final_predictions = (probabilities >= THRESHOLD).astype(int)

# 輸出結果
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': final_predictions,
})

submission.to_csv('submission.csv', index=False)

print("完成！已生成解答檔案 'submission.csv'")
print(f"測試集總人數: {len(submission)} 人")
print(f"生存人數: {(submission['Survived'] == 1).sum()} 人")
print(f"遇難人數: {(submission['Survived'] == 0).sum()} 人")
print("=" * 40)