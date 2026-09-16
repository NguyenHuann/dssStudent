import pandas as pd
import numpy as np

TRAIN_FILE = "dataset/train/train.csv"
TEST_FILE = "dataset/test/test.csv"
DATAFILE = "dataset/dataset_cleaned.csv"

TRAIN_LABELED_FILE = "dataset/train/train_labeled.csv"
TEST_LABELED_FILE = "dataset/test/test_labeled.csv"
DATA_LABELED_FILE = "dataset/dataset_labeled.csv"


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
    MAX_GRADE = 10.0
    grade_risk = 1.0 - (df["avg_grade"] / MAX_GRADE)

    avg_pass_rate = (df["1st_sem_pass_rate"] + df["2nd_sem_pass_rate"]) / 2
    pass_rate_risk = 1.0 - avg_pass_rate

    performance_risk = (0.6 * grade_risk) + (0.4 * pass_rate_risk)

    # 3. Failure Risk (Dựa trên tổng môn trượt)
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
    perf_risk, fail_risk, trend_risk, fin_risk = calculate_risk_indicators(df)

    warning_score = (
            0.40 * perf_risk +
            0.30 * fail_risk +
            0.20 * trend_risk +
            0.10 * fin_risk
    )

    # Nếu tổng số môn đăng ký cả năm = 0 -> Đẩy thẳng rủi ro lên 1.0 (Cao nhất)
    total_enrolled = df["1st_sem_enrolled"] + df["2nd_sem_enrolled"]

    warning_score = np.where(total_enrolled == 0, 1.0, warning_score)

    return warning_score


def apply_warning_labels():
    # 1. ĐỌC DỮ LIỆU (Thêm biến full_df)
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    full_df = pd.read_csv(DATAFILE)

    # 2. TÍNH WARNING SCORE CHO CẢ 3 TẬP
    train_df["WarningScore"] = calculate_warning_score(train_df)
    test_df["WarningScore"] = calculate_warning_score(test_df)
    full_df["WarningScore"] = calculate_warning_score(full_df)

    # 3. TÌM NGƯỠNG (THRESHOLDS) CHỈ TRÊN TẬP TRAIN
    threshold_low = train_df["WarningScore"].quantile(0.333)
    threshold_high = train_df["WarningScore"].quantile(0.667)

    def get_level(row):
        score = row["WarningScore"]
        avg_grade = row["avg_grade"]

        # Phân loại theo phân vị thuật toán
        if score <= threshold_low:
            label = "LOW"
        elif score <= threshold_high:
            label = "MEDIUM"
        else:
            label = "HIGH"

        # LUẬT NGHIỆP VỤ: Điểm TB dưới 5.0 không được phép nằm ở mức LOW
        if label == "LOW" and avg_grade < 5.0:
            return "MEDIUM"

        return label

    # 4. GÁN NHÃN CHO CẢ 3 TẬP (Dùng apply với axis=1 để truyền cả row)
    train_df["WarningLevel"] = train_df.apply(get_level, axis=1)
    test_df["WarningLevel"] = test_df.apply(get_level, axis=1)
    full_df["WarningLevel"] = full_df.apply(get_level, axis=1)

    # 5. LƯU DỮ LIỆU MỚI
    train_df.to_csv(TRAIN_LABELED_FILE, index=False)
    test_df.to_csv(TEST_LABELED_FILE, index=False)
    full_df.to_csv(DATA_LABELED_FILE, index=False)

    # BÁO CÁO KẾT QUẢ
    print("Phân bố WarningLevel - TRAIN:")
    print(train_df["WarningLevel"].value_counts())

    print("\nPhân bố WarningLevel - TEST:")
    print(test_df["WarningLevel"].value_counts())

    print("\nPhân bố WarningLevel - TOÀN BỘ DATASET:")
    print(full_df["WarningLevel"].value_counts())

    print("\nĐã tạo thành công các file dữ liệu có nhãn!")


if __name__ == "__main__":
    apply_warning_labels()