import os
os.system("pip install scikit-learn")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

print("🚢 鐵達尼號正式實戰：開始讀取桌面資料...")

# =============================================================================
# 1. 讀取你放在桌面的檔案
# =============================================================================
# 使用相對路徑，或是你可以改成你電腦的絕對路徑（例如：'C:/Users/陳品賢/Desktop/train.csv'）
# 如果檔案就在桌面，且你在 Spyder 的工作目錄對齊桌面，直接寫檔名即可
try:
    train_df = pd.read_csv('C:/Users/陳品賢/Desktop/train.csv')
    test_df = pd.read_csv('C:/Users/陳品賢/Desktop/test.csv')
except FileNotFoundError:
    print("❌ 找不到檔案！請確認 train.csv 和 test.csv 是否都放在桌面喔！")

# =============================================================================
# 2. 特徵工程：清理課本與考卷（兩份都要清理！）
# =============================================================================
# 我們要鎖定的五個關鍵特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# 【處理課本 train_df】
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) # 補年齡空缺
train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})  # 文字轉數字

# 【處理考卷 test_df】（考卷也要用一模一樣的規則清理，AI 才看懂）
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())   # 補年齡空缺
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})    # 文字轉數字

# 另外，考卷裡有一位客艙等級欄位有缺，我們順手幫他補上最常見的數值，防止報錯
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# 分裝特徵與答案
X_train = train_df[features]
y_train = train_df['Survived'] # 課本的正確答案
X_test = test_df[features]     # 正式考卷的題目（沒有答案）

# =============================================================================
# 3. 訓練 AI 模型（讓它研讀整本課本）
# =============================================================================
print("🤖 AI 正在研讀整本課本中...")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# =============================================================================
# 4. 正式考試：預測考卷答案並輸出成答案卡
# =============================================================================
print("📝 考試中！AI 正在預測 test.csv 裡 418 位乘客的生死...")
final_predictions = model.predict(X_test)

# 製作符合 Kaggle 格式的答案卡（包含 PassengerId 與我們的預測結果 Survived）
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_predictions
})

# 將答案卡輸出成 CSV 檔，直接存到你的桌面
submission_path = 'C:/Users/陳品賢/Desktop/my_submission.csv'
submission.to_csv(submission_path, index=False)

print("-" * 40)
print("🎉 恭喜成功！正式答案卡已經生成在你的桌面了！")
print("檔案名稱為：my_submission.csv")
print("你可以打開它看看，裡面就是 AI 幫你寫好的 418 題答案喔！")
print("-" * 40)