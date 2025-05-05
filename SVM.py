import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# 1. Đọc dữ liệu
df = pd.read_csv('DuLieuSauKhiXuLyThanhCong.csv')

print(df.columns)


# 2. Giả sử cột 'label' là nhãn, còn lại là đặc trưng
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình
model = SVC(kernel='linear', probability=True)  # 'probability=True' để dùng được ROC AUC
model.fit(X_train, y_train)

# 5. Dự đoán
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # lấy xác suất để tính AUC

# 6. In các chỉ số đánh giá
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Tính ROC AUC (chỉ dùng được cho nhị phân hoặc cần xử lý đặc biệt cho đa lớp)
if len(y.unique()) == 2:
    # Trường hợp nhị phân
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
    print("ROC AUC Score:", roc_auc)
else:
    # Trường hợp đa lớp
    y_bin = LabelBinarizer().fit_transform(y_test)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr')
    print("ROC AUC Score (multi-class):", roc_auc)
