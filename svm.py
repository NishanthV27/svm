

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)


y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

print("🔹 Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("🔹 RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

print("\nClassification Report (Linear SVM):\n", classification_report(y_test, y_pred_linear))
print("\nClassification Report (RBF SVM):\n", classification_report(y_test, y_pred_rbf))


param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n✅ Best Parameters (RBF):", grid_search.best_params_)
print("✅ Best Cross-Validation Accuracy:", grid_search.best_score_)


scores_linear = cross_val_score(svm_linear, X, y, cv=5)
scores_rbf = cross_val_score(svm_rbf, X, y, cv=5)

print("\n📊 Cross-Validation Results:")
print("Linear SVM CV Accuracy:", scores_linear.mean())
print("RBF SVM CV Accuracy:", scores_rbf.mean())


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)


svm_rbf_2d = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf_2d.fit(X_reduced, y)

def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_rbf_2d, X_reduced, y, "SVM with RBF Kernel (PCA-Reduced 2D)")
