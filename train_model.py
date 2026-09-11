import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

print("="*65)
print("  HUẤN LUYỆN VÀ SO SÁNH HIỆU NĂNG 4 MÔ HÌNH MACHINE LEARNING  ")
print("="*65)

# 1. Đọc dữ liệu đã chuyển đổi
df = pd.read_csv('dataset/danh_sach_sinh_vien.csv')

feature_cols = ['GPA', 'ChuyenCan', 'DiemGiuaKy', 'SoMonTruot', 'SoTinChi', 'LichSuCanhBao', 'NoHocPhi']
X = df[feature_cols]
y = df['NguyCo']

# 2. Chia tập Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu cho Logistic Regression và KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Khởi tạo 4 mô hình (Đã bổ sung class_weight='balanced')
models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
}
# 4. Huấn luyện và đánh giá từng mô hình
results = []
trained_models = {}

for name, model in models.items():
    if name in ['Logistic Regression', 'KNN (k=5)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    trained_models[name] = model

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'Mô hình': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)': round(prec * 100, 2),
        'Recall (%)': round(rec * 100, 2),
        'F1-Score (%)': round(f1 * 100, 2)
    })

# 5. Hiển thị bảng kết quả so sánh
df_results = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False)
print("\n" + df_results.to_string(index=False))

# 6. Đóng gói mô hình tối ưu Random Forest
rf_best = trained_models['Random Forest']
y_pred_rf = rf_best.predict(X_test)

data_pack = {
    'model': rf_best,
    'feature_cols': feature_cols,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf, labels=rf_best.classes_),
    'classes': list(rf_best.classes_),
    'report': classification_report(y_test, y_pred_rf, output_dict=True),
    'results_table': df_results
}

joblib.dump(data_pack, 'dataset/mo_hinh_nguy_co.pkl')
print("\n" + "="*65)
print("✓ ĐÃ ĐÓNG GÓI MÔ HÌNH RANDOM FOREST TỐI ƯU VÀO 'mo_hinh_nguy_co.pkl'!")
print("="*65)