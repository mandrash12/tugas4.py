import pandas as pd
from sklearn.model_selection import train_test_split

print(">>> [INFO] Library berhasil diimpor.")

print("\n>>> [LANGKAH 0] Memuat dan membagi data dari 'processed_kelulusan.csv'...")

# Menggunakan Pilihan A: Membaca file yang sudah diproses
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print("--- File 'processed_kelulusan.csv' berhasil dimuat. ---")
except FileNotFoundError:
    print("\n!!! ERROR: File 'processed_kelulusan.csv' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan script dari tugas sebelumnya.")
    exit()

# Pisahkan kembali antara Fitur (X) dan Target (y)
X = df.drop('Lulus', axis=1)
y = df['Lulus']
print("--- Data berhasil dipisahkan menjadi Fitur (X) dan Target (y). ---")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Verifikasi hasil akhir pembagian
print("\n--- Hasil Akhir Pembagian Dataset ---")
print(f"Bentuk X_train (Data Latih): \t{X_train.shape}")
print(f"Bentuk X_val (Data Validasi):\t{X_val.shape}")
print(f"Bentuk X_test (Data Uji): \t{X_test.shape}")

print("\n>>> [SELESAI] Langkah 0 berhasil. Data siap untuk diproses.")


# LANGKAH 2: PIPELINE & BASELINE RANDOM FOREST
# ===================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

print("\n>>> [LANGKAH 2] Membangun pipeline dan baseline model Random Forest...")

# 1. Mengidentifikasi kolom numerik (semua fitur kita adalah numerik)
num_cols = X_train.select_dtypes(include="number").columns

# 2. Membuat pipeline untuk preprocessing
#    - SimpleImputer: Mengisi data kosong (praktik terbaik).
#    - StandardScaler: Menyamakan skala semua fitur.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols)
    ],
    remainder="drop"
)

# 3. Mendefinisikan model Random Forest sebagai baseline
rf_baseline = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

# 4. Menggabungkan preprocessor dan model ke dalam satu pipeline utama
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", rf_baseline)
])

# 5. Melatih pipeline dengan data training
pipe.fit(X_train, y_train)
print("--- Baseline Random Forest (pipeline) berhasil dilatih. ---")

# 6. Mengevaluasi performa baseline pada data validasi
print("\n--- Mengevaluasi Baseline Model pada Data Validasi ---")
y_val_pred = pipe.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average="macro")
report_val = classification_report(y_val, y_val_pred, digits=3, zero_division=0)

print(f"Baseline RF - F1(val): {f1_val:.3f}")
print("Laporan Klasifikasi (Validasi):")
print(report_val)

print("\n>>> [SELESAI] Langkah 2 berhasil.")


# ===================================================================
# LANGKAH 3: VALIDASI SILANG
# ===================================================================
from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n>>> [LANGKAH 3] Menjalankan Validasi Silang pada baseline model...")

# 1. Menentukan strategi validasi silang (3-Fold karena keterbatasan data)
#    PERBAIKAN: n_splits diubah dari 5 menjadi 3.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


scores = cross_val_score(
    pipe, 
    X_train,
    y_train,
    cv=skf,
    scoring="f1_macro",
    n_jobs=-1
)

# 3. Menampilkan hasil
print(f"Skor F1 di setiap fold: {scores}")
print(f"CV F1-macro (Train): {scores.mean():.3f} +/- {scores.std():.3f}")

print("\n>>> [SELESAI] Langkah 3 berhasil.")

# ===================================================================
# LANGKAH 4: TUNING RINGKAS (GRIDSEARCH)
# ===================================================================
from sklearn.model_selection import GridSearchCV

print("\n>>> [LANGKAH 4] Mencari hyperparameter terbaik dengan GridSearchCV...")

# 1. Menentukan hyperparameter yang akan diuji
#    Ini adalah 'peta' kombinasi yang akan dicoba oleh GridSearchCV
param_grid = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}


gs = GridSearchCV(
    pipe, # Menggunakan pipeline baseline yang sama
    param_grid=param_grid,
    cv=skf, # Menggunakan strategi CV dari Langkah 3
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1 # Menampilkan log proses pencarian
)

# 3. Menjalankan pencarian pada data training
gs.fit(X_train, y_train)

# 4. Menampilkan hasil tuning
print("\n--- Hasil GridSearchCV ---")
print(f"Parameter terbaik: {gs.best_params_}")
print(f"Skor F1 CV terbaik: {gs.best_score_:.3f}")

# 5. Mengambil model terbaik dan mengevaluasinya pada data validasi
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
f1_val_best = f1_score(y_val, y_val_best, average="macro")

print(f"\nBest RF - F1(val): {f1_val_best:.3f}")

print("\n>>> [SELESAI] Langkah 4 berhasil.")


# ===================================================================
# LANGKAH 5: EVALUASI AKHIR (TEST SET)
# ===================================================================
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

print("\n>>> [LANGKAH 5] Melakukan evaluasi akhir pada Test Set...")

# 1. Model final adalah model terbaik dari GridSearchCV
final_model = best_model
print("--- Model final yang digunakan: Best Tuned Random Forest ---")

# 2. Membuat prediksi pada data uji
y_test_pred = final_model.predict(X_test)

# 3. Evaluasi dengan metrik standar
print("\n--- Hasil Evaluasi pada Test Set ---")
f1_test = f1_score(y_test, y_test_pred, average="macro")
report_test = classification_report(y_test, y_test_pred, digits=3, zero_division=0)

print(f"F1(test): {f1_test:.3f}")
print("Classification Report (Test):")
print(report_test)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

# 4. Evaluasi dengan ROC-AUC dan Precision-Recall
if hasattr(final_model, 'predict_proba'):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    # ROC-AUC
    try:
        auc_test = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC-AUC(test): {auc_test:.3f}")

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_test:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)'); plt.legend();
        plt.savefig("roc_test_rf.png", dpi=120)
        plt.show()
    except Exception as e:
        print(f"Tidak bisa membuat plot ROC-AUC: {e}")

    # Precision-Recall Curve
    try:
        prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, label='PR Curve')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test Set)'); plt.legend();
        plt.savefig("pr_test_rf.png", dpi=120)
        plt.show()
    except Exception as e:
        print(f"Tidak bisa membuat plot PR-Curve: {e}")

print("\n>>> [SELESAI] Langkah 5 berhasil.")


# ===================================================================
# LANGKAH 6: PENTINGNYA FITUR (FEATURE IMPORTANCE)
# ===================================================================
import numpy as np
from sklearn.inspection import permutation_importance

print("\n>>> [LANGKAH 6] Menganalisis pentingnya fitur...")

# --- 6a: Feature Importance Bawaan dari Random Forest ---
print("\n--- 6a. Feature Importance Bawaan (Gini Importance) ---")
try:
    # Mengambil nama fitur setelah diproses oleh preprocessor
    feature_names = final_model.named_steps["pre"].get_feature_names_out()
    # Mengambil nilai importance dari model di dalam pipeline
    importances = final_model.named_steps["clf"].feature_importances_
    
    # Mengurutkan fitur dari yang paling penting
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("Top 5 Fitur Paling Penting:")
    print(forest_importances.head(5))

    # Membuat plot
    fig, ax = plt.subplots(figsize=(10, 6))
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Pentingnya Fitur (Bawaan)")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

except Exception as e:
    print(f"Tidak bisa mendapatkan feature importance bawaan: {e}")

# --- 6b: Permutation Importance (Opsional) ---
print("\n--- 6b. Permutation Importance (pada Data Validasi) ---")
try:
    r = permutation_importance(
        final_model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Menampilkan hasil
    for i in r.importances_mean.argsort()[::-1]:
        print(f"{X_val.columns[i]:<20} "
              f"{r.importances_mean[i]:.3f} "
              f"+/- {r.importances_std[i]:.3f}")
except Exception as e:
     print(f"Tidak bisa menjalankan permutation importance: {e}")

print("\n>>> [SELESAI] Langkah 6 berhasil.")

# ===================================================================
print("\n>>> [SELESAI] SELURUH PROSES MACHINE LEARNING TELAH BERAKHIR! 🎉")
# ===================================================================

# ===================================================================
# LANGKAH 7: SIMPAN MODEL FINAL
# ===================================================================
import joblib

print("\n>>> [LANGKAH 7] Menyimpan model final yang sudah di-tuning...")

# 'best_model' adalah variabel dari Langkah 4 yang berisi pipeline terbaik
# Kita menyimpannya ke dalam file bernama 'rf_model.pkl'
joblib.dump(best_model, "rf_model.pkl")

print("--- Model berhasil disimpan sebagai 'rf_model.pkl' ---")


try:
    mdl = joblib.load("rf_model.pkl")
    print(">>> Model 'rf_model.pkl' berhasil dimuat.")
except FileNotFoundError:
    print("!!! ERROR: File 'rf_model.pkl' tidak ditemukan. Jalankan script utama terlebih dahulu.")
    exit()

sample_data = {
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,      
    "IPK_x_Study": 3.4 * 7      
}

# 3. Konversi data sampel menjadi DataFrame
sample_df = pd.DataFrame([sample_data])
print("\n--- Data Sampel yang Akan Diprediksi ---")
print(sample_df)

# 4. Lakukan prediksi
prediction = mdl.predict(sample_df)
prediction_result = int(prediction[0]) # Ambil hasil prediksi (0 atau 1)

# 5. Tampilkan hasil
print("\n--- Hasil Prediksi ---")
if prediction_result == 1:
    print(f"Prediksi: {prediction_result} -> Mahasiswa diprediksi LULUS.")
else:
    print(f"Prediksi: {prediction_result} -> Mahasiswa diprediksi TIDAK LULUS.")
