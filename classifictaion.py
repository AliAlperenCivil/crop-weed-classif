import cv2 as cv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from skimage.feature import graycomatrix, graycoprops, hog


# =====================================================
# 1. FEATURE ÇIKARMA
# =====================================================
def extract_features(image_path):
    img = cv.imread(image_path)
    if img is None:
        return None

    img = cv.resize(img, (128, 128))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Renk özellikleri
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / (128 * 128)

    # Kontur özellikleri
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    area = sum(cv.contourArea(c) for c in contours)

    # GLCM (doku)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # HOG (kenar)
    hog_feat = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    hog_mean = np.mean(hog_feat)

    return [
        h_mean, s_mean, v_mean,
        green_ratio,
        contour_count, area,
        contrast, homogeneity,
        hog_mean
    ]


# =====================================================
# 2. VERİ SETİ YÜKLEME (TRAIN / TEST AYRI)
# =====================================================
dataset_path = "C:/Users/alial/Downloads/DERSLER/Tasarim Calismasi 1/archive/agri_data/dataset_final"
class_names = ["crop", "weed"]

X_train, y_train = [], []
X_test, y_test = [], []

for label, class_name in enumerate(class_names):

    train_folder = os.path.join(dataset_path, "train", class_name)
    test_folder = os.path.join(dataset_path, "test", class_name)

    for file in os.listdir(train_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            feat = extract_features(os.path.join(train_folder, file))
            if feat is not None:
                X_train.append(feat)
                y_train.append(label)

    for file in os.listdir(test_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            feat = extract_features(os.path.join(test_folder, file))
            if feat is not None:
                X_test.append(feat)
                y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Train görüntü sayısı: {len(X_train)}")
print(f"Test görüntü sayısı: {len(X_test)}")


# =====================================================
# 3. MODELLER
# =====================================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# =====================================================
# 4. 5-FOLD CROSS VALIDATION (SADECE RANDOM FOREST)
# =====================================================
print("\nRandom Forest 5-Fold Cross Validation:")
rf_scores = cross_val_score(
    models["Random Forest"],
    X_train,
    y_train,
    cv=5
)

print("Fold doğrulukları:", rf_scores)
print(f"Ortalama doğruluk: %{rf_scores.mean()*100:.2f}")


# =====================================================
# 5. EĞİTİM + TEST + CONFUSION MATRIX (TEK SAYFA)
# =====================================================
num_models = len(models)
rows = math.ceil(num_models / 2)

fig, axes = plt.subplots(rows, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (model_name, model) in enumerate(models.items()):

    print(f"\n{model_name} eğitiliyor...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{model_name} Doğruluk: %{acc*100:.2f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[idx]
    )

    axes[idx].set_title(model_name)
    axes[idx].set_xlabel("Tahmin Edilen")
    axes[idx].set_ylabel("Gerçek")

# Boş subplot varsa sil
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# =====================================================
# 6. TEK GÖRÜNTÜ TESTİ
# =====================================================
def classify_single_image(image_path, model):
    feat = extract_features(image_path)
    if feat is None:
        return "Okunamadı"

    feat = np.array(feat).reshape(1, -1)
    return class_names[model.predict(feat)[0]]


test_image = os.path.join(
    dataset_path,
    "test",
    "weed",
    "agri_0_9727.jpeg"
)

print("\nTek görüntü testi:")
for model_name, model in models.items():
    print(f"{model_name}: {classify_single_image(test_image, model)}")
