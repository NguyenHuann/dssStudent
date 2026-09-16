import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

# CẤU HÌNH TRANG STREAMLIT

st.set_page_config(page_title="Hệ thống Cảnh báo Học vụ & XAI", page_icon="🎓", layout="wide")

# Trọng số TOPSIS
WEIGHTS = {"Performance": 0.40, "Failure": 0.30, "Trend": 0.20, "Financial": 0.10}

FEATURES = [
    "Debtor", "Tuition fees up to date",
    "1st_sem_enrolled", "1st_sem_evaluations", "1st_sem_approved",
    "1st_sem_grade", "1st_sem_without_evaluations", "1st_sem_failed", "1st_sem_pass_rate",
    "2nd_sem_enrolled", "2nd_sem_evaluations", "2nd_sem_approved",
    "2nd_sem_grade", "2nd_sem_without_evaluations", "2nd_sem_failed", "2nd_sem_pass_rate",
    "total_failed", "avg_grade"
]


# 1. TẢI DỮ LIỆU & MÔ HÌNH (CÓ CACHE)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/dataset_labeled.csv")
    df.insert(0, "Mã SV", [f"SV{str(i).zfill(4)}" for i in range(1, len(df) + 1)])
    return df


@st.cache_resource
def load_model_and_explainer():
    # Tải mô hình Random Forest
    model = joblib.load("models/rf_warning_classifier.pkl")
    # Khởi tạo SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    return model, explainer

# CÁC HÀM XỬ LÝ TOPSIS (GIỮ NGUYÊN)

def calculate_individual_risks(row):
    # 1. Kiểm tra Ghost Student (Không đăng ký tín chỉ nào)
    total_enrolled = row["1st_sem_enrolled"] + row["2nd_sem_enrolled"]
    if total_enrolled == 0:
        return {"Performance": 1.0, "Failure": 1.0, "Trend": 1.0, "Financial": 1.0}

    # 2. Sinh viên đi học bình thường
    financial_risk = 1.0 if (row["Debtor"] == 1 or row["Tuition fees up to date"] == 0) else 0.0
    grade_risk = max(0.0, 1.0 - (row["avg_grade"] / 10.0))
    pass_rate_risk = 1.0 - ((row["1st_sem_pass_rate"] + row["2nd_sem_pass_rate"]) / 2)
    performance_risk = (0.6 * grade_risk) + (0.4 * pass_rate_risk)
    failure_risk = min(1.0, row["total_failed"] / 10.0)
    pass_rate_trend = row["1st_sem_pass_rate"] - row["2nd_sem_pass_rate"]
    trend_risk = pass_rate_trend if pass_rate_trend > 0 else 0.0

    return {"Performance": performance_risk, "Failure": failure_risk, "Trend": trend_risk, "Financial": financial_risk}

def rank_risk_factors(risks):
    weighted_risks = {factor: value * WEIGHTS[factor] for factor, value in risks.items()}
    return sorted(weighted_risks.items(), key=lambda item: item[1], reverse=True)


def get_recommendation_rules(factor):
    rules = {
        "Failure": [
            "Ưu tiên đăng ký học lại ngay các học phần tiên quyết.",
            "Giảm số lượng tín chỉ đăng ký học mới trong học kỳ tiếp theo để tập trung trả nợ môn.",
            "Tận dụng học kỳ hè để đẩy nhanh tiến độ học lại."
        ],
        "Performance": [
            "Đăng ký tham gia các lớp phụ đạo, chuỗi ôn tập do đoàn khoa tổ chức.",
            "Lên danh sách các học phần có điểm thấp để cải thiện, kéo GPA lên mức an toàn.",
            "Xếp lịch gặp cố vấn học tập định kỳ."
        ],
        "Trend": [
            "Thiết lập lại thời gian biểu, cân bằng giữa việc học và làm thêm.",
            "Liên hệ Phòng Tư vấn học đường nếu đang gặp áp lực tâm lý.",
            "Theo dõi sát sao điểm quá trình, không đợi đến điểm thi cuối kỳ."
        ],
        "Financial": [
            "Liên hệ khẩn cấp Phòng Tài chính để xin gia hạn nộp học phí.",
            "Tìm hiểu hồ sơ xin quỹ học bổng vượt khó hoặc vay vốn sinh viên."
        ]
    }
    return rules.get(factor, [])

# 2. GIAO DIỆN CHÍNH

def main():
    df = load_data()
    model, explainer = load_model_and_explainer()

    st.sidebar.title("Menu Điều Hướng")
    menu = st.sidebar.radio("Chọn chức năng:", ["Tổng quan chung", "Chi tiết & Tư vấn cá nhân"])

    if menu == "Tổng quan chung":
        render_dashboard(df, model, explainer)
    else:
        render_student_detail(df, model, explainer)

# 3. TRANG TỔNG QUAN (CÓ SHAP GLOBAL)

def render_dashboard(df, model, explainer):
    st.title("Tổng quan Tình trạng Học tập Sinh viên")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số Sinh viên", len(df))
    col2.metric("🟢 Cảnh báo LOW", len(df[df["WarningLevel"] == "LOW"]))
    col3.metric("🟡 Cảnh báo MEDIUM", len(df[df["WarningLevel"] == "MEDIUM"]))
    col4.metric("🔴 Cảnh báo HIGH", len(df[df["WarningLevel"] == "HIGH"]))

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Phân bố Mức độ Cảnh báo")
        fig_pie = px.pie(df, names='WarningLevel', color='WarningLevel',
                         color_discrete_map={'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}, hole=0.4)
        st.plotly_chart(fig_pie, width="stretch")

    with col_chart2:
        st.subheader("Tương quan Điểm & Môn trượt")
        fig_scatter = px.scatter(df, x="avg_grade", y="total_failed", color="WarningLevel",
                                 color_discrete_map={'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'})
        st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("---")
    st.subheader("Trí tuệ nhân tạo (SHAP) - Phân tích diện rộng")
    st.markdown("Biểu đồ thể hiện mức độ đóng góp trung bình của từng đặc trưng vào quyết định cảnh báo của AI.")

    X_sample = df[FEATURES].sample(min(300, len(df)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # 1. Khởi tạo kích thước khung vẽ
    plt.figure(figsize=(10, 5))

    # 2. Để SHAP vẽ trực tiếp lên plt
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, class_names=model.classes_)

    # 3. Dùng plt.gcf() để lấy biểu đồ hiện tại đưa cho Streamlit
    st.pyplot(plt.gcf())

    # 4. Xóa bộ nhớ canvas để không bị đè hình khi tải lại trang
    plt.clf()


# 4. TRANG CHI TIẾT (CÓ SHAP LOCAL + TOPSIS)

def render_student_detail(df, model, explainer):
    st.title("Tra cứu & Tư vấn Học tập Cá nhân")
    st.markdown("---")

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        filter_level = st.selectbox("Lọc theo mức cảnh báo:", ["Tất cả", "HIGH", "MEDIUM", "LOW"])
    filtered_df = df if filter_level == "Tất cả" else df[df["WarningLevel"] == filter_level]

    if filtered_df.empty:
        st.warning(f"Không có sinh viên nào trong tập dữ liệu thuộc mức cảnh báo {filter_level}.")
        return  # Dừng render các phần bên dưới để tránh lỗi rỗng

    with col_filter2:
        student_id = st.selectbox("Chọn Mã Sinh Viên:", filtered_df["Mã SV"].tolist())

    if student_id is None:
        return

    student_data = df[df["Mã SV"] == student_id].iloc[0]

    st.subheader(f"Hồ sơ Sinh viên: {student_data['Mã SV']}")
    level = student_data['WarningLevel']

    if level == "HIGH":
        st.error("MỨC CẢNH BÁO: CAO (HIGH RISK)")
    elif level == "MEDIUM":
        st.warning("MỨC CẢNH BÁO: TRUNG BÌNH (MEDIUM RISK)")
    else:
        st.success("MỨC CẢNH BÁO: THẤP (LOW RISK) - Tình trạng an toàn")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Điểm TB (avg_grade)", round(student_data['avg_grade'], 2))
    c2.metric("Tổng môn trượt", student_data['total_failed'])
    c3.metric("Tỷ lệ đạt HK1", f"{student_data['1st_sem_pass_rate'] * 100:.1f}%")
    c4.metric("Tỷ lệ đạt HK2", f"{student_data['2nd_sem_pass_rate'] * 100:.1f}%")

    # KHỐI 1: EXPLAINABILITY (SHAP)

    st.markdown("---")
    st.subheader("Lý giải Dự đoán của AI (SHAP Explainer)")

    X_student = student_data[FEATURES].to_frame().T
    shap_values_student = explainer.shap_values(X_student)

    # Xác định index của lớp được dự đoán
    class_idx = list(model.classes_).index(level)

    # Xử lý tương thích phiên bản SHAP (list hoặc numpy array)
    if isinstance(shap_values_student, list):
        contribs = shap_values_student[class_idx][0]
    else:
        contribs = shap_values_student[0, :, class_idx]

    # Hàm định dạng hiển thị giá trị thực tế
    def format_feature_value(feat_name, val):
        if 'pass_rate' in feat_name:
            return f"{val * 100:.1f}%"
        elif isinstance(val, (float, np.floating)):
            return f"{val:.2f}"
        return str(val)


    feature_labels_with_values = [
        f"{f} ({format_feature_value(f, student_data[f])})" for f in FEATURES
    ]

    # Tạo DataFrame để vẽ bằng Plotly (Sử dụng nhãn mới có chứa giá trị)
    shap_df = pd.DataFrame({'Đặc trưng': feature_labels_with_values, 'Mức độ tác động': contribs})
    shap_df = shap_df.reindex(shap_df['Mức độ tác động'].abs().sort_values(ascending=True).index)

    # Điều chỉnh màu và nhãn động theo level
    if level == "HIGH":
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(
            lambda x: 'Tăng mức rủi ro' if x > 0 else 'Kéo giảm rủi ro')
        color_map = {'Tăng mức rủi ro': '#dc3545', 'Kéo giảm rủi ro': '#0d6efd'}  # Đỏ / Xanh dương
    elif level == "LOW":
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(lambda x: 'Tăng độ an toàn' if x > 0 else 'Kéo về rủi ro')
        color_map = {'Tăng độ an toàn': '#28a745', 'Kéo về rủi ro': '#ffc107'}  # Xanh lá / Vàng
    else:  # MEDIUM
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(
            lambda x: 'Đẩy lên mức rủi ro' if x > 0 else 'Kéo về an toàn')
        color_map = {'Đẩy lên mức rủi ro': '#ffc107', 'Kéo về an toàn': '#28a745'}  # Vàng / Xanh lá

    fig_shap = px.bar(shap_df, x='Mức độ tác động', y='Đặc trưng', orientation='h',
                      color='Màu', color_discrete_map=color_map,
                      title=f"Đóng góp của các đặc trưng vào quyết định {level}")
    fig_shap.update_layout(height=400, showlegend=True)

    st.plotly_chart(fig_shap, width="stretch")

    # KHỐI 2: ACTIONABILITY (TOPSIS)

    if level in ["HIGH", "MEDIUM"]:
        st.markdown("---")
        st.subheader("Đề xuất Lộ trình Hành động (TOPSIS Ranking)")
        st.markdown("*Dựa trên 4 nhóm rủi ro nghiệp vụ, đây là các vấn đề cần xử lý trước tiên:*")

        risks = calculate_individual_risks(student_data)
        ranked_factors = rank_risk_factors(risks)
        top_2_factors = [ranked_factors[0][0], ranked_factors[1][0]]

        factor_name_vn = {
            "Failure": "Khối lượng nợ học phần (Môn trượt)",
            "Performance": "Chất lượng học tập (Điểm & Tỷ lệ đạt)",
            "Trend": "Phong độ học tập sa sút",
            "Financial": "Vấn đề Tài chính / Học phí"
        }

        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            st.info(f"**ƯU TIÊN 1: {factor_name_vn[top_2_factors[0]]}**")
            for rule in get_recommendation_rules(top_2_factors[0]):
                st.markdown(f"- {rule}")

        with col_adv2:
            st.warning(f"**ƯU TIÊN 2: {factor_name_vn[top_2_factors[1]]}**")
            for rule in get_recommendation_rules(top_2_factors[1]):
                st.markdown(f"- {rule}")


if __name__ == "__main__":
    main()