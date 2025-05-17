import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier

from preprocess.anemia.preprocess import changeLabels, scaleData
from model_pytorch import MLP, train_mlp_model

# === Dane ===
name = os.path.basename(__file__)[:-3]
df = changeLabels()
features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaleData(X_train)
X_test_scaled = scaleData(X_test)

# === XGBoost ===
xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, f"{name}_xgb.pkl")

# === PyTorch MLP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = MLP(input_dim=X_train_scaled.shape[1], hidden_dim=64, output_dim=len(np.unique(y_train))).to(device)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
train_mlp_model(mlp_model, train_loader, criterion, optimizer, device, epochs=50)

torch.save(mlp_model.state_dict(), f"{name}_mlp.pth")

# === Ensemble ===
xgb_probs = xgb_model.predict_proba(X_test_scaled)
mlp_model.eval()
with torch.no_grad():
    mlp_probs = F.softmax(mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)), dim=1).cpu().numpy()

ensemble_probs = (xgb_probs + mlp_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)

accuracy = accuracy_score(y_test, ensemble_preds)
print(f"🎯 Ensemble Accuracy: {accuracy:.4f}")

# === ROC & Confusion Matrix ===
classes = xgb_model.classes_
y_test_bin = label_binarize(y_test, classes=classes)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], ensemble_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
class_names = ['Aplastyczna', 'Hemolityczna', 'Makrocytarna', 'Mikrocytarna',
               'Normocytarna', 'Healthy', 'Trombocytopenia']
subset = [class_names[i] for i in classes]
colors = plt.get_cmap('tab10')

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors(i),
             label=f'{subset[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class (Ensemble)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve_ensemble.png")

cm = confusion_matrix(y_test, ensemble_preds, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=subset)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix (Ensemble)')
plt.tight_layout()
plt.savefig("confusion_matrix_ensemble.png")
plt.show()
