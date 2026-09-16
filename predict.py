import pandas as pd
import joblib
import os

# Đặt tên file dữ liệu sinh viên mới của bạn ở đây (chưa có nhãn)
INPUT_FILE = "data/data_demo.csv"
# File đầu ra sau khi AI đã dự đoán xong
OUTPUT_FILE = "data/data_demo_predicted.csv"
# Đường dẫn mô hình AI đã huấn luyện
MODEL_FILE = "models/rf_warning_classifier.pkl"

# 18 đặc trưng bắt buộc AI cần để dự đoán
FEATURES = [
    "Debtor", "Tuition fees up to date",
    "1st_sem_enrolled", "1st_sem_evaluations", "1st_sem_approved",
    "1st_sem_grade", "1st_sem_without_evaluations", "1st_sem_failed", "1st_sem_pass_rate",
    "2nd_sem_enrolled", "2nd_sem_evaluations", "2nd_sem_approved",
    "2nd_sem_grade", "2nd_sem_without_evaluations", "2nd_sem_failed", "2nd_sem_pass_rate",
    "total_failed", "avg_grade"
]


def predict_and_label():
    print("=" * 50)
    print("HỆ THỐNG GÁN NHÃN CẢNH BÁO HỌC VỤ TỰ ĐỘNG (AI INFERENCE)")
    print("=" * 50)

    # 1. KIỂM TRA DỮ LIỆU ĐẦU VÀO
    if not os.path.exists(INPUT_FILE):
        print(f"LỖI: Không tìm thấy file dữ liệu mới tại '{INPUT_FILE}'.")
        print("Vui lòng chuẩn bị file CSV chứa danh sách sinh viên cần đánh giá.")
        return

    print(f"Đang tải dữ liệu từ: {INPUT_FILE}...")
    new_df = pd.read_csv(INPUT_FILE)

    # Kiểm tra xem có đủ 18 cột đặc trưng không
    missing_cols = [col for col in FEATURES if col not in new_df.columns]
    if missing_cols:
        print(f"LỖI: Dataset mới bị thiếu các cột sau: {missing_cols}")
        return

    # 2. TẢI MÔ HÌNH AI
    if not os.path.exists(MODEL_FILE):
        print(f"LỖI: Không tìm thấy mô hình AI tại '{MODEL_FILE}'. Bạn cần chạy train.py trước.")
        return

    print("Đang nạp bộ não AI (Random Forest)...")
    model = joblib.load(MODEL_FILE)

    # 3. TIẾN HÀNH DỰ ĐOÁN (INFERENCE)
    print("AI đang tiến hành chẩn đoán và gán nhãn...")
    X_new = new_df[FEATURES]

    # AI tự động phân loại LOW, MEDIUM, HIGH
    predictions = model.predict(X_new)
    new_df["WarningLevel"] = predictions

    # 4. CHỐT CHẶN AN TOÀN (HARD BUSINESS RULES)
    # Vì AI dựa trên xác suất thống kê, ta vẫn áp dụng một lớp rule cứng để
    # đảm bảo 100% không bỏ lọt các ca đặc biệt (như điểm liệt hoặc bỏ học).
    def apply_safety_rules(row):
        label = row["WarningLevel"]
        total_enrolled = row["1st_sem_enrolled"] + row["2nd_sem_enrolled"]
        avg_grade = row["avg_grade"]

        # Bắt "Sinh viên bóng ma" (Đăng ký 0 môn)
        if total_enrolled == 0:
            return "HIGH"

        # Bắt luật điểm liệt (Vét đĩa qua môn nhưng GPA < 2.0 ở hệ 4)
        if label == "LOW" and avg_grade < 2.0:
            return "MEDIUM"

        return label

    new_df["WarningLevel"] = new_df.apply(apply_safety_rules, axis=1)

    # 5. LƯU KẾT QUẢ VÀ BÁO CÁO
    new_df.to_csv(OUTPUT_FILE, index=False)

    print("\nHOÀN TẤT! Đã gán nhãn thành công.")
    print(f"File kết quả được lưu tại: {OUTPUT_FILE}")

    print("\nBẢNG THỐNG KÊ PHÂN BỐ CẢNH BÁO (DỰ ĐOÁN):")
    dist = new_df["WarningLevel"].value_counts()
    for level, count in dist.items():
        print(f" - Mức {level}: {count} sinh viên ({count / len(new_df) * 100:.1f}%)")


if __name__ == "__main__":
    predict_and_label()