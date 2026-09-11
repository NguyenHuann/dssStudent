import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from dss_engine import topsis_de_xuat

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ Hỗ Trợ Quyết Định Cảnh Báo Sớm - DSS",
    page_icon="🎓",
    layout="wide"
)

# Tải dữ liệu và mô hình AI
@st.cache_data
def load_data():
    return pd.read_csv('dataset/danh_sach_sinh_vien.csv')

df_sv = load_data()

try:
    data_pack = joblib.load('dataset/mo_hinh_nguy_co.pkl')
    model = data_pack['model']
    feature_cols = data_pack['feature_cols']
except FileNotFoundError:
    st.error("Chưa tìm thấy file 'mo_hinh_nguy_co.pkl'. Hãy chạy train_model.py trước!")
    st.stop()

# Tiêu đề chính ứng dụng
st.title("HỆ HỖ TRỢ QUYẾT ĐỊNH CẢNH BÁO SỚM HỌC TẬP (ENTERPRISE DSS)")
st.caption("Hệ thống tích hợp Machine Learning (Random Forest), Explainable AI (SHAP) & Ra quyết định TOPSIS")
st.markdown("---")

# TẠO 3 TAB GIAO DIỆN CHUYÊN NGHIỆP
tab1, tab2, tab3 = st.tabs([
    "Tab 1: Tổng Quan & Cảnh Báo Hàng Loạt",
    "Tab 2: Chẩn Đoán & Ra Quyết Định Cá Nhân",
    "Tab 3: Hiệu Năng Mô Hình AI"
])

# ==================== TAB 1: TỔNG QUAN & BATCH ANALYTICS ====================
with tab1:
    st.subheader("Thống Kê Phân Bố Nguy Cơ Học Tập Toàn Khóa")
    col_t1, col_t2 = st.columns([1, 2])
    
    counts = df_sv['NguyCo'].value_counts()
    with col_t1:
        st.metric("Tổng số sinh viên theo dõi", f"{len(df_sv):,} SV")
        st.metric("Nguy cơ Cao (Cần can thiệp gấp)", f"{counts.get('Cao', 0):,} SV", delta="Cảnh báo", delta_color="inverse")
        st.metric("Nguy cơ Trung Bình", f"{counts.get('Trung Binh', 0):,} SV")
        st.metric("Nguy cơ Thấp (An toàn)", f"{counts.get('Thap', 0):,} SV")
        
    with col_t2:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        colors = {'Cao': '#FF4B4B', 'Trung Binh': '#FFA500', 'Thap': '#4CAF50'}
        pie_colors = [colors.get(k, '#888888') for k in counts.index]
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=90)
        ax1.set_title("Tỷ lệ phân bố mức độ rủi ro học tập")
        st.pyplot(fig1)
        
    st.markdown("---")
    st.subheader("Lọc Danh Sách Sinh Viên Cần Can Thiệp")
    selected_risk = st.selectbox("Chọn mức nguy cơ muốn lọc:", ["Tất cả", "Cao", "Trung Binh", "Thap"])
    
    if selected_risk != "Tất cả":
        filtered_df = df_sv[df_sv['NguyCo'] == selected_risk]
    else:
        filtered_df = df_sv

    st.dataframe(filtered_df, use_container_width=True)
    
    # Xuất báo cáo CSV
    csv_data = filtered_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("Tải Danh Sách Báo Cáo (CSV)", csv_data, "danh_sach_canh_bao_sv.csv", "text/csv")

# ==================== TAB 2: CHẨN ĐOÁN CÁ NHÂN & TOPSIS ====================
with tab2:
    st.sidebar.header("Tùy Chọn Hồ Sơ Sinh Viên")
    input_mode = st.sidebar.radio("Nguồn dữ liệu:", ["Chọn từ CSDL (4.424 SV)", "Nhập thủ công"])
    
    if input_mode == "Chọn từ CSDL (4.424 SV)":
        search_ma = st.sidebar.selectbox("Chọn Mã SV:", df_sv['MaSV'])
        sv = df_sv[df_sv['MaSV'] == search_ma].iloc[0]
        
        ma_sv, ten_sv = sv['MaSV'], sv['TenSV']
        gpa, chuyen_can = float(sv['GPA']), int(sv['ChuyenCan'])
        diem_gk, so_mon_truot = float(sv['DiemGiuaKy']), int(sv['SoMonTruot'])
        so_tin_chi, lich_su_cb = int(sv['SoTinChi']), int(sv['LichSuCanhBao'])
        no_hp = int(sv['NoHocPhi'])
    else:
        ma_sv = st.sidebar.text_input("Mã SV:", "SV99999")
        ten_sv = st.sidebar.text_input("Tên SV:", "Nguyễn Văn A")
        gpa = st.sidebar.number_input("GPA (0.0 - 4.0):", 0.0, 4.0, 1.8, 0.1)
        chuyen_can = st.sidebar.slider("Chuyên cần (%):", 0, 100, 65)
        diem_gk = st.sidebar.number_input("Điểm giữa kỳ (0.0 - 10.0):", 0.0, 10.0, 4.5, 0.5)
        so_mon_truot = st.sidebar.number_input("Số môn trượt:", 0, 10, 2)
        so_tin_chi = st.sidebar.number_input("Số tín chỉ:", 0, 30, 18)
        lich_su_cb = st.sidebar.number_input("Lịch sử bị cảnh báo (lần):", 0, 5, 1)
        no_hp = st.sidebar.selectbox("Nợ học phí:", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")

    # Hiển thị hồ sơ chỉ số
    st.subheader(f"Hồ Sơ Chỉ Số Cá Nhân: {ten_sv} ({ma_sv})")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("GPA", f"{gpa:.2f}")
    c2.metric("Chuyên cần", f"{chuyen_can}%")
    c3.metric("Đ.Giữa kỳ", f"{diem_gk:.1f}")
    c4.metric("Môn trượt", f"{so_mon_truot}")
    c5.metric("Tín chỉ", f"{so_tin_chi}")
    c6.metric("Lịch sử CB", f"{lich_su_cb} lần")
    c7.metric("Nợ học phí", "Có" if no_hp==1 else "Không")

    st.markdown("---")
    
    # TÙY CHỈNH TRỌNG SỐ TOPSIS
    st.subheader("Cấu Hình Trọng Số Quyết Định TOPSIS (Dành Cho Cố Vấn)")
    st.caption("Điều chỉnh thanh kéo ưu tiên để hệ thống tự động tính toán lại thứ tự phương án:")
    
    wc1, wc2, wc3, wc4 = st.columns(4)
    w_hieu_qua = wc1.slider("C1: Hiệu quả can thiệp:", 0.1, 1.0, 0.4, 0.05)
    w_phu_hop = wc2.slider("C2: Mức độ phù hợp:", 0.1, 1.0, 0.3, 0.05)
    w_kha_thi = wc3.slider("C3: Tính khả thi:", 0.1, 1.0, 0.2, 0.05)
    w_chi_phi = wc4.slider("C4: Tiết kiệm chi phí:", 0.1, 1.0, 0.1, 0.05)
    
    user_weights = [w_hieu_qua, w_phu_hop, w_kha_thi, w_chi_phi]

    if st.button("Chạy Phân Tích Chẩn Đoán & Đề Xuất TOPSIS", type="primary"):
        input_df = pd.DataFrame([[gpa, chuyen_can, diem_gk, so_mon_truot, so_tin_chi, lich_su_cb, no_hp]], columns=feature_cols)
        
        # AI Dự đoán
        nguy_co_pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        idx_pred = classes.index(nguy_co_pred)
        
        col_l, col_r = st.columns([1, 1])
        
        # Cột trái: SHAP XAI
        with col_l:
            st.subheader("Dự Đoán & Giải Thích AI (SHAP)")
            if nguy_co_pred == 'Cao':
                st.error(f"**Nguy cơ:** CAO (Độ tin cậy AI: {probs[idx_pred]*100:.1f}%)")
            elif nguy_co_pred == 'Trung Binh':
                st.warning(f"⚡ **Nguy cơ:** TRUNG BÌNH (Độ tin cậy AI: {probs[idx_pred]*100:.1f}%)")
            else:
                st.success(f"**Nguy cơ:** THẤP (Độ tin cậy AI: {probs[idx_pred]*100:.1f}%)")
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            if isinstance(shap_values, list):
                sv_shap = shap_values[idx_pred][0]
            else:
                sv_shap = shap_values[0, :, idx_pred]
                
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            colors_list = ['red' if val > 0 else 'green' for val in sv_shap]
            ax2.barh(feature_cols, sv_shap, color=colors_list)
            ax2.set_xlabel('Tác động SHAP (Đỏ = Thúc đẩy nguy cơ, Xanh = Giảm)')
            ax2.set_title(f'SHAP values giải thích cho SV {ten_sv}')
            st.pyplot(fig2)

        # Cột phải: Động cơ TOPSIS
        with col_r:
            st.subheader("Đề Xuất Phương Án Can Thiệp (TOPSIS)")
            df_topsis = topsis_de_xuat(ma_sv, ten_sv, gpa, chuyen_can, so_mon_truot, so_tin_chi, nguy_co_pred, weights=user_weights)
            st.dataframe(df_topsis[['Xếp hạng', 'Phương án đề xuất', 'Điểm TOPSIS (C*)']], use_container_width=True)
            
            top_1 = df_topsis.iloc[0]['Phương án đề xuất']
            st.info(f"**KHUYẾN NGHỊ HÀNG ĐẦU:** Áp dụng phương án **'{top_1}'**.")

# ==================== TAB 3: HIỆU NĂNG MÔ HÌNH AI ====================
with tab3:
    st.subheader("📈 Báo Cáo Đánh Giá Hiệu Năng Machine Learning")
    
    # 1. Bảng so sánh 4 mô hình ở Bước 3
    if 'results_table' in data_pack:
        st.markdown("### 1. Bảng So Sánh Hiệu Năng 4 Thuật Toán ML")
        st.dataframe(data_pack['results_table'], use_container_width=True)
        st.caption("-> Mô hình Random Forest được lựa chọn đóng gói làm mô hình chính nhờ sự cân bằng giữa Accuracy và F1-Score.")
    
    st.markdown("---")
    col_m1, col_m2 = st.columns([1, 1])
    
    # 2. Confusion Matrix
    with col_m1:
        st.markdown("### 2. Ma Trận Nhầm Lẫn (Confusion Matrix)")
        cm = data_pack['confusion_matrix']
        cls_names = data_pack['classes']
        
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        cax = ax3.matshow(cm, cmap=plt.cm.Blues)
        fig3.colorbar(cax)
        
        ax3.set_xticks(range(len(cls_names)))
        ax3.set_yticks(range(len(cls_names)))
        ax3.set_xticklabels(cls_names)
        ax3.set_yticklabels(cls_names)
        
        for i in range(len(cls_names)):
            for j in range(len(cls_names)):
                ax3.text(j, i, str(cm[i, j]), va='center', ha='center', color='red' if cm[i, j] > 100 else 'black')
                
        plt.xlabel('AI Dự đoán')
        plt.ylabel('Thực tế')
        st.pyplot(fig3)
        
    # 3. Báo cáo Precision / Recall
    with col_m2:
        st.markdown("### 3. Báo Cáo Chi Tiết Của Random Forest")
        rep_df = pd.DataFrame(data_pack['report']).transpose()
        st.dataframe(rep_df.style.highlight_max(axis=0), use_container_width=True)