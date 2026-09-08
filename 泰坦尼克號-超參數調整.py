import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
# 2. 特徵工程 (Feature Engineering)
# ==========================================
print("進行特徵工程與清理...")

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

# (3) 計算家庭人數與單身標記
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# (4) 船艙與港口補全
df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# (5) 類別變數數值化 (One-Hot Encoding)
df['Sex_Code'] = df['Sex'].map({'female': 0, 'male': 1})
df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)

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
# 4. 手動設定參數、訓練模型並生成答案卡
# ==========================================
print("載入中...")

# 在這裡直接手動調整你想要測試的超參數
model = RandomForestClassifier(
# --- 1. 樹的數量與深度 ---
    n_estimators=30,                # 樹的數量（常用：50, 100, 150, 200, 300）
    max_depth=11,                  # 樹的最大深度（常用：3, 4, 5, 6, 8, None）

    # -- 2. 節點分割與防過擬合 --
    min_samples_split=22,             # 內部節點再劃分所需的最小樣本數（常用：2, 4, 6, 10）
    min_samples_leaf=8,              # 葉子節點所需的最小樣本數（常用：1, 2, 4, 8）
    max_leaf_nodes=40,             # 最大葉子節點數，限制樹的生長（例如：15, 30, 或 None 不限制）
    min_impurity_decrease=0.0,       # 節點分割所需的最小不純度減少量（例如：0.0, 0.01）

    # -- 3. 特徵與資料抽樣--
    max_features='sqrt',             # 每棵樹隨機抽取的特徵數（'sqrt', 'log2', 0.8, 或 None 取全部）
    bootstrap=True,                  # 是否使用放回抽樣（ Bagging 模式）
    max_samples=0.3,                # 當 bootstrap=True 時，每棵樹抽取的資料比例（例如：0.8 或 None 代表 100%）

    # -- 4. 評估指標與類別不平衡 --
    criterion='gini',                # 分割指標，評估不純度（可選：'gini', 'entropy', 'log_loss'）
    class_weight=None,               # 類別權重，處理資料不平衡（可選：None, 'balanced', 'balanced_subsample'）

    # -- 5. 系統效能與可重複性 --
    random_state=None,               # 固定隨機種子，確保每次執行結果完全一致
    n_jobs=1,                        # CPU 運算核心數（-1 代表調用全部核心全力加速）
    oob_score=False                  # 是否計算袋外得分（Out-of-Bag Score，用做自我驗證）
)

# 使用完整的訓練資料直接擬合
model.fit(X, y)

# 預測考卷
final_predictions = model.predict(X_test)

# 輸出 submission.csv
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': final_predictions,
})

submission.to_csv('submission.csv', index=False)

print("=" * 40)
print("完成！已於桌面生成解答檔案 'submission.csv'")
print(f"測試集總人數: {len(submission)} 人")
print(f"預測生存人數: {(submission['Survived'] == 1).sum()} 人")
print(f"預測遇難人數: {(submission['Survived'] == 0).sum()} 人")
print("=" * 40)

