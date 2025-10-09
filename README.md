import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv("kelulusan_mahasiswa.csv")
except FileNotFoundError:
    print("Error: 'kelulusan_mahasiswa.csv' not found. Please check the file path.")
    exit()

print("--- DataFrame Info ---")
df.info()
print("\n--- First 5 Rows ---")
print(df.head())


print("\n--- Missing Values Check ---")
print(df.isnull().sum())
df = df.drop_duplicates()
print(f"\nDataFrame shape after dropping duplicates: {df.shape}")


df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

print("\n--- DataFrame Descriptive Statistics ---")
print(df.describe())


print("\n--- Generating Visualizations ---")
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
sns.boxplot(x=df['IPK'])
plt.title('Boxplot of IPK')

plt.subplot(2, 3, 2)
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title('Histogram of IPK')

plt.subplot(2, 3, 3)

sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title('IPK vs. Study Time by Graduation Status')
plt.subplot(2, 3, 4)

numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Feature Correlation Heatmap')


plt.tight_layout()
plt.show() # Crucial to display the plots

df.to_csv("processed_kelulusan.csv", index=False)
print("\nProcessed data saved to 'processed_kelulusan.csv'")


X = df.select_dtypes(include=['number']).drop('Lulus', axis=1, errors='ignore')
y = df['Lulus']

if X.empty or y.empty:
    print("\nError: X or y is empty after selection. Check the 'Lulus' column and other data types.")
    exit()

print(f"\nFeatures (X) used for splitting: {list(X.columns)}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3, # 30% for temp (validation + test)
    stratify=y,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5, # 50% of X_temp/y_temp
    stratify=y_temp,
    random_state=42
)

print("\n--- Data Split Shapes ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

