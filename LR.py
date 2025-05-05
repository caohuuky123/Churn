import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# 1. Đọc dữ liệu
df = pd.read_csv('DuLieuSauKhiXuLyThanhCong.csv')
print(df.columns)

# 2. Chia đặc trưng và nhãn
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện Logistic Regression
model = LogisticRegression(max_iter=1000)  # tăng max_iter nếu cảnh báo hội tụ
model.fit(X_train, y_train)

# 5. Dự đoán
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 6. Báo cáo đánh giá
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. ROC AUC
if len(y.unique()) == 2:
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
    print("ROC AUC Score:", roc_auc)
else:
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr')
    print("ROC AUC Score (multi-class):", roc_auc)
