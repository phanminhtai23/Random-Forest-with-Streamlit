import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

data = pd.read_csv('./iris.csv')

X = data.iloc[:, :-1]  # Tất cả các cột trừ cột cuối
y = data.iloc[:, -1]   # Cột cuối cùng là nhãn
# print(X, y)


sum = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Acc ở lần train thứ %d: %.2f" % (i+1, accuracy))
    sum += accuracy
print(f"Độ chính xác trung bình 10 lần train: {round(sum/10,2)}")

dump(model, 'Random_Forest.joblib')
print("Đã lưu model `./Random_Forest`")