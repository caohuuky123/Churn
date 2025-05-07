import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# 1. Đọc dữ liệu
df = pd.read_csv('DuLieuSauKhiXuLyThanhCong.csv')

print("Các cột dữ liệu:", df.columns)

# 2. Giả sử cột 'Churn' là nhãn, còn lại là đặc trưng
if 'Churn' not in df.columns:
    raise ValueError("Không tìm thấy cột 'Churn' trong dữ liệu!")

X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Xác suất dùng cho tính AUC

# 6. In kết quả đánh giá
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Accuracy Score ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Tính và in ROC AUC Score
print("=== ROC AUC Score ===")
if len(y.unique()) == 2:
    # Nhị phân
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
    print("ROC AUC (Binary):", roc_auc)
else:
    # Đa lớp
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr')
    print("ROC AUC (Multi-class):", roc_auc)
