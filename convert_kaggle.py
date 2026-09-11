import pandas as pd
import numpy as np

print("="*60)
print("  BẮT ĐẦU CHUYỂN ĐỔI VÀ LÀM SẠCH DATASET KAGGLE  ")
print("="*60)

# 1. Đọc file CSV gốc từ Kaggle (Tự động nhận diện dấu phẩy hoặc chấm phẩy)
try:
    df_raw = pd.read_csv('dataset/data_kaggle.csv', sep=';')
    if len(df_raw.columns) == 1:
        df_raw = pd.read_csv('dataset/data_kaggle.csv', sep=',')
except Exception:
    df_raw = pd.read_csv('dataset/data_kaggle.csv')

print(f"✓ Đã đọc thành công file gốc: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột.")

# 2. Ánh xạ nhãn Target sang 3 mức nguy cơ
target_map = {
    'Dropout': 'Cao',
    'Enrolled': 'Trung Binh',
    'Graduate': 'Thap'
}
df_raw['NguyCo'] = df_raw['Target'].map(target_map)

# 3. Tạo Mã SV và Tên SV ngẫu nhiên để cá nhân hóa
n_samples = len(df_raw)
np.random.seed(42)

ho = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Vũ', 'Đặng', 'Bùi']
dem = ['Văn', 'Thị', 'Minh', 'Đức', 'Anh', 'Ngọc', 'Quang', 'Thảo']
ten = ['An', 'Bình', 'Cường', 'Dũng', 'Giang', 'Hương', 'Khánh', 'Linh', 'Nam', 'Trang']

df_raw['MaSV'] = [f"SV{20000 + i}" for i in range(1, n_samples + 1)]
df_raw['TenSV'] = [f"{np.random.choice(ho)} {np.random.choice(dem)} {np.random.choice(ten)}" for _ in range(n_samples)]

# 4. Trích xuất & Quy đổi các chỉ số học tập
# - GPA: Quy đổi cột 'Curricular units 1st sem (grade)' (thang 20) về thang 4.0
gpa_converted = np.round(df_raw['Curricular units 1st sem (grade)'] / 5.0, 2)

# - Điểm giữa kỳ: Quy đổi về thang 10.0
diem_gk_converted = np.round(df_raw['Curricular units 1st sem (grade)'] / 2.0, 1)

# - Tỷ lệ chuyên cần (%): Ước tính từ số lần tham gia đánh giá
enrolled = df_raw['Curricular units 1st sem (enrolled)']
evaluations = df_raw['Curricular units 1st sem (evaluations)']
chuyen_can_calc = np.where(enrolled > 0, np.round((evaluations / (enrolled * 3)) * 100), 85)

# - Số môn trượt = Môn đăng ký - Môn hoàn thành
so_mon_truot_calc = enrolled - df_raw['Curricular units 1st sem (approved)']

# 5. Đóng gói vào DataFrame sạch chuẩn hóa
df_clean = pd.DataFrame({
    'MaSV': df_raw['MaSV'],
    'TenSV': df_raw['TenSV'],
    'GPA': gpa_converted.clip(lower=0.0, upper=4.0),
    'ChuyenCan': pd.Series(chuyen_can_calc).clip(lower=0, upper=100).astype(int),
    'DiemGiuaKy': diem_gk_converted.clip(lower=0.0, upper=10.0),
    'SoMonTruot': pd.Series(so_mon_truot_calc).clip(lower=0).astype(int),
    'SoTinChi': (enrolled * 3).astype(int),
    'LichSuCanhBao': df_raw['Debtor'].astype(int),
    'NoHocPhi': df_raw['Debtor'].astype(int),
    'NguyCo': df_raw['NguyCo']
})

# 6. Lưu thành file CSV chính thức cho toàn bộ dự án
df_clean.to_csv('danh_sach_sinh_vien.csv', index=False, encoding='utf-8-sig')

print("="*60)
print("✓ ĐÃ TẠO THÀNH CÔNG FILE 'danh_sach_sinh_vien.csv'!")
print(f"  Tổng số sinh viên: {len(df_clean)}")
print("  Phân bố nguy cơ:")
print(df_clean['NguyCo'].value_counts())
print("="*60)