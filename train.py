import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_FILE = "dataset/train/train_labeled.csv"
TEST_FILE = "dataset/test/test_labeled.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "rf_warning_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# 18 Đặc trưng đầu vào (loại bỏ WarningScore, WarningLevel và Target)
FEATURES = [
    "Debtor", "Tuition fees up to date",
    "1st_sem_enrolled", "1st_sem_evaluations", "1st_sem_approved",
    "1st_sem_grade", "1st_sem_without_evaluations", "1st_sem_failed", "1st_sem_pass_rate",
    "2nd_sem_enrolled", "2nd_sem_evaluations", "2nd_sem_approved",
    "2nd_sem_grade", "2nd_sem_without_evaluations", "2nd_sem_failed", "2nd_sem_pass_rate",
    "total_failed", "avg_grade"
]


def train_and_evaluate():
    # ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
    print("Đang tải dữ liệu...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df[FEATURES]
    y_train = train_df["WarningLevel"]

    X_test = test_df[FEATURES]
    y_test = test_df["WarningLevel"]

    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Test: {X_test.shape}\n")

    # HUẤN LUYỆN VÀ TUNING (5-FOLD CV)

    print("Bắt đầu huấn luyện mô hình Random Forest với 5-Fold CV...")

    # Khởi tạo mô hình cơ sở
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Lưới tham số cần tinh chỉnh (Hyperparameter Tuning)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Sử dụng GridSearchCV (mặc định cv=5 là Stratified K-Fold)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',  # Dùng F1 Macro để đối xử công bằng với cả 3 nhãn Low/Medium/High
        n_jobs=-1,  # Dùng tối đa CPU
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\nTham số tối ưu tìm được: {grid_search.best_params_}")

    # Lưu mô hình
    joblib.dump(best_rf, MODEL_FILE)
    print(f"Đã lưu mô hình tại: {MODEL_FILE}\n")


    # ĐÁNH GIÁ TRÊN TẬP TEST

    print("=" * 50)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 50)

    y_pred = best_rf.predict(X_test)

    # Đảm bảo thứ tự hiển thị luôn chuẩn: LOW, MEDIUM, HIGH
    labels_order = ["LOW", "MEDIUM", "HIGH"]

    print("\n1. Báo cáo phân loại (Classification Report):")
    print(classification_report(y_test, y_pred, labels=labels_order))

    print("2. Ma trận nhầm lẫn (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"Thực tế {l}" for l in labels_order],
                         columns=[f"Dự đoán {l}" for l in labels_order])
    print(cm_df)

    # KIỂM CHỨNG THỰC NGHIỆM VỚI TARGET GỐC

    print("\n" + "=" * 50)
    print("KIỂM CHỨNG THỰC NGHIỆM VỚI 'TARGET' GỐC")
    print("=" * 50)

    # Gắn nhãn dự đoán ngược lại vào tập test để so sánh
    test_df["Predicted_WarningLevel"] = y_pred

    # Bảng chéo (Cross-tabulation) tính tỷ lệ Dropout / Enrolled / Graduate theo từng mức cảnh báo
    cross_tab = pd.crosstab(
        index=test_df["Predicted_WarningLevel"],
        columns=test_df["Target"],
        normalize='index'  # Tính theo tỷ lệ % trên mỗi hàng
    ) * 100

    # Sắp xếp lại hiển thị
    cross_tab = cross_tab.reindex(["LOW", "MEDIUM", "HIGH"])

    print("\nTỷ lệ % Target thực tế ứng với mỗi mức Cảnh báo (Dự đoán):")
    print(cross_tab.round(2).astype(str) + "%")



if __name__ == "__main__":
    train_and_evaluate()