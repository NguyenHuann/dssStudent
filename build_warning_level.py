import pandas as pd
import numpy as np

TRAIN_FILE = "dataset/train/train.csv"
TEST_FILE = "dataset/test/test.csv"

TRAIN_LABELED_FILE = "dataset/train/train_labeled.csv"
TEST_LABELED_FILE = "dataset/test/test_labeled.csv"


def calculate_risk_indicators(df):
    """
    Hàm tính toán 4 nhóm rủi ro (Risk Indicators) cho từng sinh viên.
    Tất cả các rủi ro được chuẩn hóa tương đối về khoảng [0, 1].
    Rủi ro càng gần 1 -> Nguy cơ càng cao.
    """
    # 1. Financial Risk (0: Không rủi ro, 1: Nợ học phí)
    # Giả định: Debtor=1 (Nợ), Tuition fees up to date=0 (Chưa đóng)
    financial_risk = np.where((df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0), 1.0, 0.0)

    # 2. Performance Risk (Dựa trên điểm trung bình và tỷ lệ đạt)
    # Giả định thang điểm là 20 (thường thấy ở dataset sinh viên Bồ Đào Nha). Nếu thang 10, thay bằng 10.
    MAX_GRADE = 10.0
    grade_risk = 1.0 - (df["avg_grade"] / MAX_GRADE)

    avg_pass_rate = (df["1st_sem_pass_rate"] + df["2nd_sem_pass_rate"]) / 2
    pass_rate_risk = 1.0 - avg_pass_rate

    performance_risk = (0.6 * grade_risk) + (0.4 * pass_rate_risk)

    # 3. Failure Risk (Dựa trên tổng môn trượt)
    # Chuẩn hóa tạm thời: chia cho ngưỡng môn trượt tối đa giả định (ví dụ 10 môn)
    # Có thể dùng MinMaxScaler của sklearn sau nếu muốn chặt chẽ hơn
    MAX_FAILED_EXPECTED = 10.0
    total_failed_risk = df["total_failed"] / MAX_FAILED_EXPECTED
    total_failed_risk = total_failed_risk.clip(upper=1.0)  # Đảm bảo không vượt quá 1

    failure_risk = total_failed_risk

    # 4. Performance Trend Risk (Sự suy giảm từ kỳ 1 sang kỳ 2)
    # Trend dương -> Tỷ lệ đạt kỳ 1 > kỳ 2 -> Đang sa sút
    pass_rate_trend = df["1st_sem_pass_rate"] - df["2nd_sem_pass_rate"]
    trend_risk = np.where(pass_rate_trend > 0, pass_rate_trend, 0.0)

    return performance_risk, failure_risk, trend_risk, financial_risk


def calculate_warning_score(df):
    """
    Tính WarningScore dựa trên trọng số đã thiết kế.
    """
    perf_risk, fail_risk, trend_risk, fin_risk = calculate_risk_indicators(df)

    # Áp dụng trọng số: 40% Performance, 30% Failure, 20% Trend, 10% Financial
    warning_score = (
            0.40 * perf_risk +
            0.30 * fail_risk +
            0.20 * trend_risk +
            0.10 * fin_risk
    )
    return warning_score


def apply_warning_labels():
    # 1. ĐỌC DỮ LIỆU
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # 2. TÍNH WARNING SCORE CHO CẢ 2 TẬP
    train_df["WarningScore"] = calculate_warning_score(train_df)
    test_df["WarningScore"] = calculate_warning_score(test_df)

    # 3. TÌM NGƯỠNG (THRESHOLDS) CHỈ TRÊN TẬP TRAIN
    # Cắt thành 3 phần bằng nhau dựa trên phân vị 33% và 67%
    threshold_low = train_df["WarningScore"].quantile(0.333)
    threshold_high = train_df["WarningScore"].quantile(0.667)

    print(f"Ngưỡng phân loại học được từ Train set:")
    print(f" - LOW to MEDIUM: {threshold_low:.4f}")
    print(f" - MEDIUM to HIGH: {threshold_high:.4f}\n")

    # Hàm gán nhãn
    def get_level(score):
        if score <= threshold_low:
            return "LOW"
        elif score <= threshold_high:
            return "MEDIUM"
        else:
            return "HIGH"

    # 4. GÁN NHÃN CHO CẢ TẬP TRAIN VÀ TEST
    train_df["WarningLevel"] = train_df["WarningScore"].apply(get_level)
    test_df["WarningLevel"] = test_df["WarningScore"].apply(get_level)

    # 5. LƯU DỮ LIỆU MỚI
    train_df.to_csv(TRAIN_LABELED_FILE, index=False)
    test_df.to_csv(TEST_LABELED_FILE, index=False)

    # BÁO CÁO KẾT QUẢ
    print("Phân bố WarningLevel - TRAIN:")
    print(train_df["WarningLevel"].value_counts())

    print("\nPhân bố WarningLevel - TEST:")
    print(test_df["WarningLevel"].value_counts())

    print("\nĐã tạo thành công các file dữ liệu có nhãn!")


if __name__ == "__main__":
    apply_warning_labels()