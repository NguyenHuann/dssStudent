import numpy as np
import pandas as pd

def topsis_de_xuat(ma_sv, ten_sv, gpa, chuyen_can, so_mon_truot, so_tin_chi, nguy_co, weights=[0.4, 0.3, 0.2, 0.1]):
    """
    Thuật toán TOPSIS xếp hạng các giải pháp can thiệp học tập
    - nguy_co: Mức nguy cơ AI dự đoán ('Thap', 'Trung Binh', 'Cao')
    - weights: Trọng số [C1: Hiệu quả, C2: Phù hợp, C3: Khả thi, C4: Tiết kiệm chi phí]
    """
    # 1. Trường hợp nguy cơ Thấp: Duy trì lộ trình hiện tại
    if nguy_co == 'Thap':
        return pd.DataFrame([{
            'Mã SV': ma_sv,
            'Tên SV': ten_sv,
            'Phương án đề xuất': 'Tiếp tục duy trì kế hoạch học tập hiện tại',
            'Điểm TOPSIS (C*)': 1.0,
            'Xếp hạng': 1
        }])

    # 2. Danh sách 4 phương án đề xuất can thiệp cho nguy cơ Trung Binh / Cao
    phuong_an_names = [
        'Gặp trực tiếp cố vấn học tập',
        'Giảm số tín chỉ đăng ký kỳ tới',
        'Học lại / củng cố các môn yếu',
        'Xây dựng kế hoạch học tập cá nhân'
    ]
    
    # Ma trận quyết định ban đầu X (4 phương án x 4 tiêu chí)
    # Tiêu chí: [C1: Hiệu quả (Càng cao càng tốt), C2: Phù hợp, C3: Khả thi, C4: Tiết kiệm]
    X = np.array([
        [9.0, 8.0, 7.0, 8.0],
        [8.0, 8.0, 9.0, 9.0],
        [9.0, 9.0, 8.0, 7.0],
        [8.0, 9.0, 10.0, 9.0]
    ])

    # Điều chỉnh ma trận theo ngữ cảnh đặc thù của sinh viên
    if chuyen_can < 65:
        X[0, 1] = 10.0 # Tăng độ phù hợp của việc gặp cố vấn
    if so_tin_chi >= 20:
        X[1, 1] = 10.0 # Tăng độ phù hợp của việc giảm tín chỉ
    if so_mon_truot >= 2:
        X[2, 1] = 10.0 # Tăng độ phù hợp của việc học lại môn yếu

    # 3. Chuẩn hóa trọng số (Tổng trọng số = 1)
    w = np.array(weights) / np.sum(weights)

    # 4. Chuẩn hóa ma trận quyết định (Vector Normalization)
    norm_factors = np.sqrt(np.sum(X**2, axis=0))
    R = X / norm_factors
    V = R * w

    # 5. Xác định giải pháp lý tưởng dương (A+) và lý tưởng âm (A-)
    A_plus = np.max(V, axis=0)
    A_minus = np.min(V, axis=0)

    # 6. Tính khoảng cách Euclidean
    S_plus = np.sqrt(np.sum((V - A_plus)**2, axis=1))
    S_minus = np.sqrt(np.sum((V - A_minus)**2, axis=1))

    # 7. Tính độ tương tự tương đối C*
    denom = S_plus + S_minus
    C_star = np.where(denom == 0, 0, S_minus / denom)

    # 8. Đóng gói kết quả bảng xếp hạng
    df_res = pd.DataFrame({
        'Mã SV': ma_sv,
        'Tên SV': ten_sv,
        'Phương án đề xuất': phuong_an_names,
        'Điểm TOPSIS (C*)': np.round(C_star, 4)
    }).sort_values(by='Điểm TOPSIS (C*)', ascending=False).reset_index(drop=True)
    
    df_res['Xếp hạng'] = df_res.index + 1
    return df_res