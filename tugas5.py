import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

print(">>> [INFO] Semua library berhasil diimpor.")

# ===================================================================
# LANGKAH 1: MUAT DAN SIAPKAN DATA
# ===================================================================
print("\n>>> [LANGKAH 1] Memuat dan membagi data dari 'processed_kelulusan.csv'...")

try:
    df = pd.read_csv("processed_kelulusan.csv")
except FileNotFoundError:
    print("\n!!! ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    exit()

# Pisahkan Fitur (X) dan Target (y)
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Lakukan pembagian dataset (dengan perbaikan)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"--- Data berhasil dimuat dan dibagi. Ukuran X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape} ---")


# ===================================================================
# LANGKAH 2: BASELINE MODEL & PIPELINE (LOGISTIC REGRESSION)
# ===================================================================
print("\n>>> [LANGKAH 2] Membangun baseline model dengan Pipeline...")

# Definisikan preprocessor sekali saja, untuk digunakan di semua model
num_cols = X_train.select_dtypes(include="number").columns
preprocessor = ColumnTransformer(
    transformers=[("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)],
    remainder="drop"
)

# Definisikan model Regresi Logistik
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# Buat pipeline untuk Regresi Logistik
pipe_lr = Pipeline([("pre", preprocessor), ("clf", logreg)])
pipe_lr.fit(X_train, y_train)

# Evaluasi baseline model pada data validasi
print("--- Mengevaluasi Baseline Model pada Data Validasi ---")
y_val_pred = pipe_lr.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average="macro")
report_val = classification_report(y_val, y_val_pred, digits=3, zero_division=0)
print(f"Baseline (LogReg) F1(val): {f1_val:.3f}")
print("Laporan Klasifikasi (Validasi) Baseline:")
print(report_val)


# ===================================================================
# LANGKAH 3: MODEL ALTERNATIF (RANDOM FOREST)
# ===================================================================
print("\n>>> [LANGKAH 3] Membangun model alternatif dengan Random Forest...")

# Definisikan model Random Forest
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42)

# Buat pipeline baru untuk Random Forest (menggunakan preprocessor yang sama)
pipe_rf = Pipeline([("pre", preprocessor), ("clf", rf)])
pipe_rf.fit(X_train, y_train)

# Evaluasi model Random Forest pada data validasi
y_val_pred_rf = pipe_rf.predict(X_val)
f1_val_rf = f1_score(y_val, y_val_pred_rf, average="macro")
print(f"--- Mengevaluasi Model Alternatif pada Data Validasi ---")
print(f"RandomForest F1(val): {f1_val_rf:.3f}")


# ===================================================================
# LANGKAH 4: VALIDASI SILANG & TUNING RINGKAS (GRIDSEARCHCV)
# ===================================================================
print("\n>>> [LANGKAH 4] Mencari hyperparameter terbaik dengan GridSearchCV...")

# Definisikan strategi validasi silang (5-Fold)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Definisikan hyperparameter yang akan diuji untuk Random Forest
param_grid = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

# Siapkan GridSearchCV
gs = GridSearchCV(pipe_rf, param_grid=param_grid, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)

# Jalankan pencarian
gs.fit(X_train, y_train)

# Tampilkan hasil tuning
print("\n--- Hasil GridSearchCV ---")
print(f"Parameter terbaik: {gs.best_params_}")
print(f"Skor F1 CV terbaik: {gs.best_score_:.3f}")

# Evaluasi model terbaik hasil tuning pada data validasi
best_rf = gs.best_estimator_
y_val_pred_best = best_rf.predict(X_val)
f1_val_best = f1_score(y_val, y_val_pred_best, average="macro")
print(f"\nBest Tuned RF F1(val): {f1_val_best:.3f}")


# ===================================================================
print("\n>>> [SELESAI] Semua proses modeling telah dijalankan! ✅")
# ===================================================================


# ===================================================================
# LANGKAH 5: EVALUASI AKHIR (TEST SET)
# ===================================================================
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print("\n>>> [LANGKAH 5] Melakukan evaluasi akhir pada Test Set...")

# 1. Memilih model final (terbaik dari langkah sebelumnya)
final_model = best_rf
print("--- Model final yang digunakan: Best Tuned Random Forest ---")

# 2. Membuat prediksi pada data uji (X_test)
y_test_pred = final_model.predict(X_test)

# 3. Mengevaluasi dengan metrik standar
print("\n--- Hasil Evaluasi pada Test Set ---")
f1_test = f1_score(y_test, y_test_pred, average="macro")
report_test = classification_report(y_test, y_test_pred, digits=3)

print(f"F1(test): {f1_test:.3f}")
print("Classification Report (Test):")
print(report_test)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))


if hasattr(final_model, 'predict_proba'):
    # Mengambil probabilitas untuk kelas positif (kelas '1')
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    try:
        # Menghitung AUC Score
        auc_test = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC-AUC(test): {auc_test:.3f}")

        # Membuat plot Kurva ROC
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Garis acuan
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) - Test Set')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120) # Menyimpan plot sebagai file
        print("--- Plot Kurva ROC berhasil disimpan sebagai 'roc_test.png' ---")
        plt.show()

    except Exception as e:
        print(f"Tidak bisa membuat plot ROC-AUC: {e}")
