import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 載入資料與特徵工程
# ==========================================
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
os.chdir(desktop_path)

print("正在讀取資料...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df = pd.concat([train, test], sort=False).reset_index(drop=True)

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

df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Fare'] = df.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex_Code'] = df['Sex'].map({'female': 0, 'male': 1})
df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)

train_df = df[df['Survived'].notnull()].copy()
features = [
    c
    for c in train_df.columns
    if c not in ['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin']
]

X = train_df[features]
y = train_df['Survived'].astype(int)

# ==========================================
# 2. 資料集切割 (80% Train, 20% Val)
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# ==========================================
# 3. 調整並精細化 9 個超參數的級距範圍
# ==========================================
param_range_dict = {
    # 增加小數量的點 (1~50) 觀察前期變化
    'n_estimators': [5, 10, 20, 30, 50, 80, 100, 150, 200, 300],
    # 級距加密為 1~15
    'max_depth': list(range(1, 16)),
    # 縮小起點與加大終點 (3~50)
    'max_leaf_nodes': [2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
    # 加大上限至 40，觀察完全平緩的過程
    'min_samples_split': [2, 4, 6, 8, 10, 14, 18, 22, 28, 35, 40],
    # 加大上限至 20
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    # 加密抽樣比例
    'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 在微小數值區間 (0.0 ~ 0.005) 進行加密
    'min_impurity_decrease': [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.005,
        0.01,
        0.015,
    ],
    # 在微小數值區間 (0.0 ~ 0.005) 進行加密
    'ccp_alpha': [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.015],
    # 改用具體的「特徵個數」(1 個特徵 ~ 全部特徵)
    'max_features': list(range(1, X.shape[1] + 1)),
}

# ==========================================
# 4. 繪製精細版 3x3 曲線圖 (含最佳點標記)
# ==========================================
print('開始計算並繪製精細版圖表...')

plt.figure(figsize=(19, 15))

for i, (param_name, param_range) in enumerate(param_range_dict.items(), 1):
    train_scores = []
    val_scores = []

    for val in param_range:
        rf_kwargs = {'random_state': 42, 'n_jobs': 1, param_name: val}
        rf = RandomForestClassifier(**rf_kwargs)

        rf.fit(X_train, y_train)

        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)

        train_scores.append(f1_score(y_train, train_pred))
        val_scores.append(f1_score(y_val, val_pred))

    plt.subplot(3, 3, i)

    # 繪製訓練集 (藍線) 與 驗證集 (紅線)
    plt.plot(
        param_range,
        train_scores,
        label='Train Score',
        color='blue',
        marker='o',
        markersize=4,
        linewidth=1.8,
    )
    plt.plot(
        param_range,
        val_scores,
        label='Val Score',
        color='red',
        marker='s',
        markersize=4,
        linewidth=1.8,
    )

    # 自動找出 Val Score 的最高點並標註紅點與數值
    best_idx = int(np.argmax(val_scores))
    best_val_param = param_range[best_idx]
    best_val_score = val_scores[best_idx]

    plt.plot(
        best_val_param,
        best_val_score,
        'ro',
        markersize=8,
        markeredgecolor='black',
        label=f'Best Val: {best_val_param}',
    )

    plt.title(f'Train vs Val: {param_name}', fontsize=11, fontweight='bold')
    plt.xlabel(param_name, fontsize=9)
    plt.ylabel('F1 Score', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()

print('\n完成！精細版 9 張圖表已輸出。')