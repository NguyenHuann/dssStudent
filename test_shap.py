import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

print("="*50)
print("  KIỂM THỬ GIẢI THÍCH MÔ HÌNH AI BẰNG SHAP  ")
print("="*50)

# 1. Tải mô hình và dữ liệu
data_pack = joblib.load('mo_hinh_nguy_co.pkl')
model = data_pack['model']
feature_cols = data_pack['feature_cols']

df = pd.read_csv('dataset/danh_sach_sinh_vien.csv')

# 2. Lấy 1 sinh viên mẫu (ví dụ: dòng đầu tiên)
sv_sample = df.iloc[0]
input_data = pd.DataFrame([sv_sample[feature_cols].values], columns=feature_cols)

# 3. Dự đoán
nguy_co_pred = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]

print(f"Sinh viên: {sv_sample['TenSV']} ({sv_sample['MaSV']})")
print(f"Mức nguy cơ AI dự đoán: {nguy_co_pred}")

# 4. Tính toán SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

classes = list(model.classes_)
idx_class = classes.index(nguy_co_pred)

if isinstance(shap_values, list):
    sv_shap = shap_values[idx_class][0]
else:
    sv_shap = shap_values[0, :, idx_class]

# 5. Vẽ biểu đồ SHAP và lưu thành ảnh
plt.figure(figsize=(7, 4))
colors = ['red' if val > 0 else 'green' for val in sv_shap]
plt.barh(feature_cols, sv_shap, color=colors)
plt.xlabel('Tác động SHAP (Đỏ = Thúc đẩy nhãn, Xanh = Giảm nhãn)')
plt.title(f'SHAP values cho SV {sv_sample["TenSV"]}')
plt.tight_layout()
plt.savefig('shap_test.png')

print("✓ Đã tính toán SHAP và lưu ảnh biểu đồ thử nghiệm vào 'shap_test.png'!")