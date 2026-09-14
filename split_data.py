import os
import pandas as pd
from sklearn.model_selection import train_test_split

# CẤU HÌNH

INPUT_FILE = "dataset/dataset_cleaned.csv"

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

TRAIN_FILE = os.path.join(TRAIN_DIR, "train.csv")
TEST_FILE = os.path.join(TEST_DIR, "test.csv")

TEST_SIZE = 0.30
RANDOM_STATE = 42

TARGET_COLUMN = "Target"

# TẠO THƯ MỤC

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ĐỌC DATASET

df = pd.read_csv(INPUT_FILE)

print("Kích thước dataset ban đầu:", df.shape)
print("\nPhân bố Target ban đầu:")
print(df[TARGET_COLUMN].value_counts())

# CHIA TRAIN / TEST

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COLUMN]
)

# LƯU FILE

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# KIỂM TRA KẾT QUẢ

print(f"Train: {len(train_df)} dòng")
print(f"Test : {len(test_df)} dòng")

print("\nPhân bố Target - Train:")
print(train_df[TARGET_COLUMN].value_counts())

print("\nPhân bố Target - Test:")
print(test_df[TARGET_COLUMN].value_counts())

print("\nĐã tạo:")
print(f"  {TRAIN_FILE}")
print(f"  {TEST_FILE}")