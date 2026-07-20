# 1. 匯入工具包與內建的鳶尾花資料
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("正在讀取資料與訓練模型...")

# 2. 讀取資料
iris = load_iris()
X = iris.data
y = iris.target

# 3. 切分資料：80% 訓練，20% 考試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 選擇模型（決策樹）
model = DecisionTreeClassifier()

# 5. 讓模型開始學習
model.fit(X_train, y_train)

# 6. 預測並計算準確度
predictions = model.predict(X_test)
print("-" * 30)
print(f"鳶尾花預測準確率: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("-" * 30)
