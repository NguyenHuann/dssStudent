import csv

INPUT_FILE = "dataset/dataset.csv"
OUTPUT_FILE = "dataset/dataset_cleaned.csv"

# Các đặc trưng cần giữ lại

FEATURES = [
    "Debtor",
    "Tuition fees up to date",

    "1st_sem_enrolled",
    "1st_sem_evaluations",
    "1st_sem_approved",
    "1st_sem_grade",
    "1st_sem_without_evaluations",
    "1st_sem_failed",
    "1st_sem_pass_rate",

    "2nd_sem_enrolled",
    "2nd_sem_evaluations",
    "2nd_sem_approved",
    "2nd_sem_grade",
    "2nd_sem_without_evaluations",
    "2nd_sem_failed",
    "2nd_sem_pass_rate",

    "total_failed",
    "avg_grade",
    "Target"
]

# Hàm chuyển dữ liệu sang số

def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Hàm tính tỷ lệ đạt

def calculate_pass_rate(approved, evaluations):
    if evaluations == 0:
        return 0.0

    return approved / evaluations

# Đọc dataset gốc và tạo dataset mới

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as infile:

    reader = csv.DictReader(infile)

    # Kiểm tra các cột cần thiết trong dataset gốc
    required_columns = [
        "Debtor",
        "Tuition fees up to date",

        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",

        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Target"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in reader.fieldnames
    ]

    if missing_columns:
        print("LỖI: Dataset thiếu các cột sau:")

        for column in missing_columns:
            print(" -", column)

        raise SystemExit

    # Tạo file CSV mới

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=FEATURES
        )

        writer.writeheader()

        # Xử lý từng sinh viên

        row_count = 0

        for row in reader:

            sem1_enrolled = to_float(
                row["Curricular units 1st sem (enrolled)"]
            )

            sem1_evaluations = to_float(
                row["Curricular units 1st sem (evaluations)"]
            )

            sem1_approved = to_float(
                row["Curricular units 1st sem (approved)"]
            )

            sem1_grade = to_float(row["Curricular units 1st sem (grade)"]) / 2.0

            sem1_without_evaluations = to_float(
                row["Curricular units 1st sem (without evaluations)"]
            )


            # Số học phần trượt HK1
            #
            # Failed = Evaluations - Approved

            sem1_failed = (
                    sem1_enrolled - sem1_approved
            )


            # Tỷ lệ đạt HK1
            sem1_pass_rate = calculate_pass_rate(
                sem1_approved,
                sem1_evaluations
            )

            sem2_enrolled = to_float(
                row["Curricular units 2nd sem (enrolled)"]
            )

            sem2_evaluations = to_float(
                row["Curricular units 2nd sem (evaluations)"]
            )

            sem2_approved = to_float(
                row["Curricular units 2nd sem (approved)"]
            )

            sem2_grade = to_float(row["Curricular units 2nd sem (grade)"]) / 2.0

            sem2_without_evaluations = to_float(
                row["Curricular units 2nd sem (without evaluations)"]
            )


            # Số học phần trượt HK2
            sem2_failed = (
                    sem2_enrolled - sem2_approved
            )


            # Tỷ lệ đạt HK2
            sem2_pass_rate = calculate_pass_rate(
                sem2_approved,
                sem2_evaluations
            )

            # ĐẶC TRƯNG TỔNG HỢP

            total_failed = (
                sem1_failed + sem2_failed
            )


            avg_grade = (
                sem1_grade + sem2_grade
            ) / 2.0

            # Tạo dòng dữ liệu mới

            new_row = {

                "Debtor": row["Debtor"],

                "Tuition fees up to date":
                    row["Tuition fees up to date"],


                "1st_sem_enrolled":
                    int(sem1_enrolled),

                "1st_sem_evaluations":
                    int(sem1_evaluations),

                "1st_sem_approved":
                    int(sem1_approved),

                "1st_sem_grade":
                    sem1_grade,

                "1st_sem_without_evaluations":
                    int(sem1_without_evaluations),

                "1st_sem_failed":
                    int(sem1_failed),

                "1st_sem_pass_rate":
                    round(sem1_pass_rate, 4),


                "2nd_sem_enrolled":
                    int(sem2_enrolled),

                "2nd_sem_evaluations":
                    int(sem2_evaluations),

                "2nd_sem_approved":
                    int(sem2_approved),

                "2nd_sem_grade":
                    sem2_grade,

                "2nd_sem_without_evaluations":
                    int(sem2_without_evaluations),

                "2nd_sem_failed":
                    int(sem2_failed),

                "2nd_sem_pass_rate":
                    round(sem2_pass_rate, 4),


                "total_failed":
                    int(total_failed),

                "avg_grade":
                    round(avg_grade, 4),
                "Target":
                    row["Target"],
            }


            writer.writerow(new_row)

            row_count += 1

# 8. Thông báo kết quả

print("=" * 60)
print("ĐÃ HOÀN THÀNH!")
print("=" * 60)

print(f"Số dòng dữ liệu: {row_count}")
print(f"Số đặc trưng: {len(FEATURES)}")
print(f"File đầu ra: {OUTPUT_FILE}")

print("\nCác đặc trưng:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:2}. {feature}")
