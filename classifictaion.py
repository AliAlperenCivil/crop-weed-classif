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


# 1) Feature extraction
def extract_features(image_path):
    img = cv.imread(image_path)
    if img is None:
        return None

    img = cv.resize(img, (128, 128))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Color features (HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / (128 * 128)

    # Contour-based features
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    area = sum(cv.contourArea(c) for c in contours)

    # Texture features (GLCM)
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

    # Edge/gradient features (HOG)
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


# 2) Dataset loading (train/test)
dataset_path = os.environ.get("DATASET_PATH", "./dataset_final")
class_names = ["crop", "weed"]  # 0=crop, 1=weed

X_train, y_train = [], []
X_test, y_test = [], []

# Keep image paths for error analysis
test_paths = []

for label, class_name in enumerate(class_names):

    train_folder = os.path.join(dataset_path, "train", class_name)
    test_folder = os.path.join(dataset_path, "test", class_name)

    for filename in os.listdir(train_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(train_folder, filename) 
            feat = extract_features(img_path)
            if feat is not None:
                X_train.append(feat)
                y_train.append(label)
                

    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_folder, filename)  
            feat = extract_features(img_path)
            if feat is not None:
                X_test.append(feat)
                y_test.append(label)
                test_paths.append(img_path) 

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# 3) Models
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


# 4) Cross-validation (Random Forest)
print("\nRandom Forest 5-Fold Cross Validation:")
rf_scores = cross_val_score(
    models["Random Forest"],
    X_train,
    y_train,
    cv=5
)

print("Fold accuracies:", rf_scores)
print(f"Mean accuracy: {rf_scores.mean()*100:.2f}%")


# 5) Training, evaluation & error analysis
num_models = len(models)
rows = math.ceil(num_models / 2)

fig, axes = plt.subplots(rows, 2, figsize=(10, 8))
axes = axes.flatten()

# Output directory for misclassified samples
out_dir = "misclassified_examples"
os.makedirs(out_dir, exist_ok=True)

for idx, (model_name, model) in enumerate(models.items()):

    print(f"\nTraining: {model_name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy ({model_name}): {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save a couple of misclassified samples (RF) for inspection
    if model_name == "Random Forest":
        wrong_idx = np.where(y_test != y_pred)[0]

        false_weed_as_crop = None  # true=weed(1), pred=crop(0)
        false_crop_as_weed = None  # true=crop(0), pred=weed(1)

        for i in wrong_idx:
            true_label = y_test[i]
            pred_label = y_pred[i]

            if true_label == 1 and pred_label == 0 and false_weed_as_crop is None:
                false_weed_as_crop = test_paths[i]

            if true_label == 0 and pred_label == 1 and false_crop_as_weed is None:
                false_crop_as_weed = test_paths[i]

            if false_weed_as_crop is not None and false_crop_as_weed is not None:
                break

        if false_weed_as_crop is not None:
            img = cv.imread(false_weed_as_crop)
            if img is not None:
                cv.imwrite(os.path.join(out_dir, "RF_weed_as_crop.jpg"), img)

        if false_crop_as_weed is not None:
            img = cv.imread(false_crop_as_weed)
            if img is not None:
                cv.imwrite(os.path.join(out_dir, "RF_crop_as_weed.jpg"), img)

        print("\n[Random Forest] Saved misclassified examples:")
        print("Weed -> Crop:", false_weed_as_crop)
        print("Crop -> Weed:", false_crop_as_weed)
        print(f"Folder: {os.path.abspath(out_dir)}")

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
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")

# Remove unused subplots
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# 6) Single image inference
def classify_single_image(image_path, model):
    feat = extract_features(image_path)
    if feat is None:
        return "Unreadable image"

    feat = np.array(feat).reshape(1, -1)
    return class_names[model.predict(feat)[0]]


test_image = os.path.join(
    dataset_path,
    "test",
    "weed",
    "agri_0_9727.jpeg"
)

print("\nSingle image test:")
for model_name, model in models.items():
    print(f"{model_name}: {classify_single_image(test_image, model)}")
        
