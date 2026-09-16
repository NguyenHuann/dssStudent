import pandas as pd
import numpy as np

# Dùng tập test đã gán nhãn
INPUT_FILE = "data/data_demo_predicted.csv"

# Trọng số của các yếu tố rủi ro (giống lúc tính Warning Score)
WEIGHTS = {
    "Performance": 0.40,
    "Failure": 0.30,
    "Trend": 0.20,
    "Financial": 0.10
}


# TÍNH TOÁN RỦI RO (RISK INDICATORS)

def calculate_individual_risks(row):
    """Tính 4 chỉ số rủi ro cho MỘT sinh viên (giá trị từ 0 đến 1)"""

    # 1. Kiểm tra Ghost Student (Không đăng ký tín chỉ nào)
    total_enrolled = row["1st_sem_enrolled"] + row["2nd_sem_enrolled"]
    if total_enrolled == 0:
        return {"Performance": 1.0, "Failure": 1.0, "Trend": 1.0, "Financial": 1.0}

    # 2. Financial Risk
    financial_risk = 1.0 if (row["Debtor"] == 1 or row["Tuition fees up to date"] == 0) else 0.0

    # 3. Performance Risk (Đã sửa thành thang điểm 4.0)
    MAX_GRADE = 4.0
    grade_risk = 1.0 - (row["avg_grade"] / MAX_GRADE)
    # Tránh giá trị âm nếu điểm có sai số
    grade_risk = max(0.0, grade_risk)

    avg_pass_rate = (row["1st_sem_pass_rate"] + row["2nd_sem_pass_rate"]) / 2
    pass_rate_risk = 1.0 - avg_pass_rate
    performance_risk = (0.6 * grade_risk) + (0.4 * pass_rate_risk)

    # 4. Failure Risk
    MAX_FAILED_EXPECTED = 10.0
    failure_risk = row["total_failed"] / MAX_FAILED_EXPECTED
    failure_risk = min(1.0, failure_risk)  # Cap ở mức 1.0

    # 5. Trend Risk
    pass_rate_trend = row["1st_sem_pass_rate"] - row["2nd_sem_pass_rate"]
    trend_risk = pass_rate_trend if pass_rate_trend > 0 else 0.0

    return {
        "Performance": performance_risk,
        "Failure": failure_risk,
        "Trend": trend_risk,
        "Financial": financial_risk
    }


# XẾP HẠNG (Mô phỏng TOPSIS Scoring)

def rank_risk_factors(risks):
    """Xếp hạng các yếu tố rủi ro dựa trên Điểm rủi ro * Trọng số"""
    weighted_risks = {}
    for factor, value in risks.items():
        weighted_risks[factor] = value * WEIGHTS[factor]

    # Sắp xếp giảm dần (Rủi ro nào có điểm trọng số cao nhất -> Ưu tiên 1)
    ranked_factors = sorted(weighted_risks.items(), key=lambda item: item[1], reverse=True)
    return ranked_factors


# RECOMMENDATION ENGINE (Rule-based)

def generate_advice(ranked_factors, row_data):
    """Tạo báo cáo lời khuyên dạng Text (Console) bằng cách gọi hàm rules"""

    report = f"BÁO CÁO TƯ VẤN HỌC TẬP\n"
    report += f"{'-' * 40}\n"
    report += f"MỨC CẢNH BÁO: {row_data['WarningLevel']} WARNING\n"
    report += f"GPA: {row_data['avg_grade']:.2f}/4.0 | Tổng số môn trượt: {row_data['total_failed']}\n"
    report += f"{'-' * 40}\n"
    report += "Hệ thống nhận thấy sinh viên đang gặp rủi ro học tập. Dưới đây là lộ trình hành động ưu tiên:\n\n"

    top_2_factors = [ranked_factors[0][0], ranked_factors[1][0]]

    for i, factor in enumerate(top_2_factors, 1):
        factor_name_vn = {
            "Failure": "Giải quyết nợ học phần (Số lượng môn trượt cao)",
            "Performance": "Cải thiện chất lượng học tập (Điểm số & Tỷ lệ đạt thấp)",
            "Trend": "Phong độ học tập sa sút",
            "Financial": "Xử lý vấn đề Tài chính / Học phí"
        }

        report += f"ƯU TIÊN {i}: {factor_name_vn[factor]}\n"

        # TỐI ƯU HÓA: Thay vì viết lại rules, ta gọi thẳng hàm get_recommendation_rules
        for advice in get_recommendation_rules(factor):
            report += f"   - {advice}\n"

    return report

def get_recommendation_rules(factor):
    """Trả về danh sách lời khuyên cụ thể cho từng yếu tố rủi ro (Dùng cho giao diện Streamlit)"""
    rules = {
        "Failure": [
            "Ưu tiên đăng ký học lại ngay các học phần tiên quyết.",
            "Giảm số lượng tín chỉ đăng ký học mới trong học kỳ tiếp theo để dồn toàn bộ thời gian trả nợ môn.",
            "Tận dụng học kỳ phụ/học kỳ hè để đẩy nhanh tiến độ học lại."
        ],
        "Performance": [
            "Đăng ký tham gia các lớp phụ đạo, chuỗi ôn tập do đoàn khoa hoặc CLB học thuật tổ chức.",
            "Lên danh sách các học phần có điểm thấp (D) nhưng dễ cải thiện để học lại, nhằm kéo GPA lên mức an toàn.",
            "Yêu cầu xếp lịch gặp trực tiếp Cố vấn học tập để rà soát lộ trình học."
        ],
        "Trend": [
            "Hệ thống ghi nhận sự sa sút rõ rệt ở học kỳ 2 so với học kỳ 1. Cần thiết lập lại thời gian biểu, cân bằng giữa việc học và làm thêm.",
            "Tự rà soát các yếu tố ngoại cảnh gây xao nhãng. Nếu gặp áp lực tâm lý, hãy liên hệ Phòng Tư vấn học đường.",
            "Theo dõi sát sao điểm quá trình trong kỳ tới, không đợi đến điểm thi cuối kỳ mới phản ứng."
        ],
        "Financial": [
            "Liên hệ khẩn cấp Phòng Tài chính để làm thủ tục xin gia hạn thời gian nộp học phí, tránh bị hủy học phần.",
            "Tìm hiểu hồ sơ xin quỹ học bổng vượt khó hoặc vay vốn sinh viên."
        ]
    }
    return rules.get(factor, [])


# CHẠY THỬ NGHIỆM TRÊN DỮ LIỆU

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Không tìm thấy file {INPUT_FILE}. Hãy chắc chắn bạn đã chạy file tạo nhãn trước.")
        return

    # Lọc ra những sinh viên bị cảnh báo HIGH
    high_risk_students = df[df["WarningLevel"] == "HIGH"].reset_index(drop=True)

    if high_risk_students.empty:
        print("Không có sinh viên nào ở mức HIGH WARNING trong tập dữ liệu.")
        return

    print(f"Đã tìm thấy {len(high_risk_students)} sinh viên ở mức HIGH. Đang xuất báo cáo mẫu...\n")
    print("=" * 60)

    # Lấy ngẫu nhiên 2 sinh viên để in báo cáo thử nghiệm
    sample_students = high_risk_students.sample(2, random_state=42)

    for idx, student_row in sample_students.iterrows():
        # Bước 1: Tính rủi ro
        risks = calculate_individual_risks(student_row)

        # Bước 2: Xếp hạng TOPSIS
        ranked_factors = rank_risk_factors(risks)

        # Bước 3: Xuất Recommendation
        report = generate_advice(ranked_factors, student_row)

        print(report)
        print("=" * 60)


if __name__ == "__main__":
    main()