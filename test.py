import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Загрузка данных из dataset.csv
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: dataset.csv not found. Make sure the file exists and the path is correct.")
    exit()

# 2. Разделение данных на обучающую и тестовую выборки
#    - train_size - доля данных, которая пойдет в обучающую выборку (например, 0.8 для 80%)
#    - random_state - для воспроизводимости результатов (можно выбрать любое число)
train_df, test_df = train_test_split(df, train_size=0.8, random_state=42)

# 3. Сохранение обучающей выборки в train.csv
try:
    train_df.to_csv("train.csv", index=False)  # index=False чтобы не сохранять индекс
    print("train.csv created successfully.")
except Exception as e:
    print(f"Error saving train.csv: {e}")

# 4. Сохранение тестовой выборки в test.csv
try:
    test_df.to_csv("test.csv", index=False)  # index=False чтобы не сохранять индекс
    print("test.csv created successfully.")
except Exception as e:
    print(f"Error saving test.csv: {e}")