import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

from topsis_recommendation import calculate_individual_risks, rank_risk_factors, get_recommendation_rules

# CẤU HÌNH TRANG STREAMLIT
st.set_page_config(page_title="Hệ thống Cảnh báo Học vụ & XAI", page_icon="🎓", layout="wide")

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
    df = pd.read_csv("data/data_demo_predicted.csv")
    df.insert(0, "Mã SV", [f"SV{str(i).zfill(4)}" for i in range(1, len(df) + 1)])
    return df


@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("models/rf_warning_classifier.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer


# 2. GIAO DIỆN CHÍNH
def main():
    df = load_data()
    model, explainer = load_model_and_explainer()

    st.sidebar.title("Menu Điều Hướng")
    menu = st.sidebar.radio("Chọn chức năng:", [
        "Tổng quan chung",
        "Chi tiết & Tư vấn cá nhân",
        "Dự đoán Sinh viên Mới"
    ])

    if menu == "Tổng quan chung":
        render_dashboard(df, model, explainer)
    elif menu == "Chi tiết & Tư vấn cá nhân":
        render_student_detail(df, model, explainer)
    else:
        render_prediction_page(model, explainer)


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

    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, class_names=model.classes_)
    st.pyplot(plt.gcf())
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
        return

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

    st.markdown("---")
    st.subheader("Lý giải Dự đoán của AI (SHAP Explainer)")
    st.markdown(f"*Tại sao sinh viên này lại bị xếp vào mức **{level}**? Yếu tố nào đóng vai trò cốt lõi?*")

    X_student = student_data[FEATURES].to_frame().T
    shap_values_student = explainer.shap_values(X_student)
    class_idx = list(model.classes_).index(level)

    if isinstance(shap_values_student, list):
        contribs = shap_values_student[class_idx][0]
    else:
        contribs = shap_values_student[0, :, class_idx]

    def format_feature_value(feat_name, val):
        if 'pass_rate' in feat_name:
            return f"{val * 100:.1f}%"
        elif isinstance(val, (float, np.floating)):
            return f"{val:.2f}"
        return str(val)

    # 1. BẢNG TỪ ĐIỂN DỊCH TÊN ĐẶC TRƯNG SANG TIẾNG VIỆT
    FEATURE_NAMES_VN = {
        "Debtor": "Đang nợ nần/khoản vay",
        "Tuition fees up to date": "Đã hoàn thành học phí",

        "1st_sem_enrolled": "Số môn đăng ký HK1",
        "1st_sem_evaluations": "Số lượt đi thi HK1",
        "1st_sem_approved": "Số môn đạt HK1",
        "1st_sem_grade": "Điểm trung bình HK1",
        "1st_sem_without_evaluations": "Môn không đánh giá HK1",
        "1st_sem_failed": "Số môn trượt HK1",
        "1st_sem_pass_rate": "Tỷ lệ qua môn HK1",

        "2nd_sem_enrolled": "Số môn đăng ký HK2",
        "2nd_sem_evaluations": "Số lượt đi thi HK2",
        "2nd_sem_approved": "Số môn đạt HK2",
        "2nd_sem_grade": "Điểm trung bình HK2",
        "2nd_sem_without_evaluations": "Môn không đánh giá HK2",
        "2nd_sem_failed": "Số môn trượt HK2",
        "2nd_sem_pass_rate": "Tỷ lệ qua môn HK2",

        "total_failed": "Tổng số môn trượt cả năm",
        "avg_grade": "Điểm TB tích lũy (GPA)"
    }

    # 2. GHÉP TÊN TIẾNG VIỆT VỚI GIÁ TRỊ THỰC TẾ
    # Sử dụng .get(f, f) để an toàn: Nếu không tìm thấy tên dịch, nó sẽ giữ nguyên tên gốc
    feature_labels_with_values = [
        f"{FEATURE_NAMES_VN.get(f, f)} ({format_feature_value(f, student_data[f])})"
        for f in FEATURES
    ]

    # 3. TẠO DATAFRAME VÀ VẼ BIỂU ĐỒ (Dùng nhãn tiếng Việt)
    shap_df = pd.DataFrame({'Đặc trưng': feature_labels_with_values, 'Mức độ tác động': contribs})
    shap_df = shap_df.reindex(shap_df['Mức độ tác động'].abs().sort_values(ascending=True).index)
    if level == "HIGH":
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(lambda x: 'Tăng mức rủi ro' if x > 0 else 'Kéo giảm rủi ro')
        color_map = {'Tăng mức rủi ro': '#dc3545', 'Kéo giảm rủi ro': '#0d6efd'}
    elif level == "LOW":
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(lambda x: 'Tăng độ an toàn' if x > 0 else 'Kéo về rủi ro')
        color_map = {'Tăng độ an toàn': '#28a745', 'Kéo về rủi ro': '#ffc107'}
    else:
        shap_df['Màu'] = shap_df['Mức độ tác động'].apply(lambda x: 'Đẩy lên mức rủi ro' if x > 0 else 'Kéo về an toàn')
        color_map = {'Đẩy lên mức rủi ro': '#ffc107', 'Kéo về an toàn': '#28a745'}

    fig_shap = px.bar(shap_df, x='Mức độ tác động', y='Đặc trưng', orientation='h',
                      color='Màu', color_discrete_map=color_map,
                      title=f"Đóng góp của các đặc trưng vào quyết định {level}")
    fig_shap.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_shap, width="stretch")

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


# 5. TRANG DỰ ĐOÁN SINH VIÊN MỚI (MỚI TÍCH HỢP)
def render_prediction_page(model, explainer):
    st.title("Dự đoán Nhóm Cảnh báo cho Sinh viên Mới")
    st.markdown("---")
    st.markdown(
        "Tải lên tệp CSV chứa danh sách sinh viên mới (chưa có nhãn). Hệ thống sẽ tự động sử dụng AI để chẩn đoán và phân loại mức độ rủi ro.")

    uploaded_file = st.file_uploader("Chọn tệp CSV dữ liệu sinh viên:", type=["csv"])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)

            # Kiểm tra xem có đủ cột đặc trưng không
            missing_cols = [col for col in FEATURES if col not in new_df.columns]
            if missing_cols:
                st.error(f"Tệp CSV bị thiếu các cột đặc trưng bắt buộc sau: {missing_cols}")
                return

            st.success("Đã tải tệp thành công! Đang tiến hành phân tích...")

            # Dự đoán bằng mô hình AI
            predictions = model.predict(new_df[FEATURES])
            new_df["WarningLevel"] = predictions

            # Áp dụng luật an toàn nghiệp vụ cứng
            def apply_safety_rules(row):
                label = row["WarningLevel"]
                total_enrolled = row["1st_sem_enrolled"] + row["2nd_sem_enrolled"]
                avg_grade = row["avg_grade"]

                if total_enrolled == 0:
                    return "HIGH"
                if label == "LOW" and avg_grade < 2.0:
                    return "MEDIUM"
                return label

            new_df["WarningLevel"] = new_df.apply(apply_safety_rules, axis=1)

            # Thêm mã sinh viên trực quan
            if "Mã SV" not in new_df.columns:
                new_df.insert(0, "Mã SV", [f"SV{str(i).zfill(4)}" for i in range(1, len(new_df) + 1)])

            st.markdown("---")
            st.subheader("Kết quả Dự đoán Tổng quan")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tổng số Sinh viên đánh giá", len(new_df))
            col2.metric("🟢 LOW", len(new_df[new_df["WarningLevel"] == "LOW"]))
            col3.metric("🟡 MEDIUM", len(new_df[new_df["WarningLevel"] == "MEDIUM"]))
            col4.metric("🔴 HIGH", len(new_df[new_df["WarningLevel"] == "HIGH"]))

            st.markdown("---")
            st.subheader("Bảng Chi tiết Kết quả Dự đoán")
            st.dataframe(new_df, use_container_width=True)

            # Nút tải file kết quả
            csv_data = new_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải xuống Kết quả (CSV)",
                data=csv_data,
                file_name="sinh_vien_moi_da_gan_nhan.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý tệp: {e}")


if __name__ == "__main__":
    main()