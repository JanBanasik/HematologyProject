# === train_pipeline.py (Modyfikacja - Usunięcie Trombocytopenii) ===
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# Usunięto import PCA
# from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier

# === Setup ===
# Ustaw ścieżkę do NOWEGO wygenerowanego pliku (anemia_only)
data_path = "../../generatingData/anemia/synthetic_data_card_parameters_anemia_only_generated_vae.csv"
preprocess_dir = "preprocess/anemia"
os.makedirs(preprocess_dir, exist_ok=True)

script_name = os.path.basename(__file__)[:-3]

# === Label mapping (USUNIĘTO TROMBOCYTOPENIĘ) ===
# Kolejność numerycznych etykiet zostanie zachowana dla pozostałych klas
label_mapping = {
    "Anemia Mikrocytarna": 0,
    "Anemia Makrocytarna": 1,
    "Anemia Hemolityczna": 2,
    "Anemia Aplastyczna": 3,
    "Anemia Normocytarna": 4,
    # Usunięto: "Trombocytopenia": 5, # Ta etykieta numeryczna jest teraz wolna
    "Healthy": 5 # Healthy przyjmuje teraz etykietę numeryczną 5
    # Pamiętaj, że jeśli dodasz więcej klas, musisz je ponumerować dalej (6, 7...)
}

# === DEFINE THE FULL SET OF EXPECTED NUMERICAL CLASSES ===
# Ta lista będzie teraz zawierać numeryczne etykiety dla 6 klas
FULL_NUMERICAL_CLASSES = sorted(label_mapping.values()) # Powinno być [0, 1, 2, 3, 4, 5]
print(f"Pełny zestaw oczekiwanych klas numerycznych: {FULL_NUMERICAL_CLASSES}")


# === Load & preprocess ===
print(f"Ładowanie danych z: {data_path}")
df = pd.read_csv(data_path)

# === Identify features and labels ===
label_col_name = 'Label'
# Lista cech jest zgodna z kolumnami w pliku (25 cech)
features = [col for col in df.columns.tolist() if col != label_col_name] # Automatycznie pobiera kolumny z pliku

# Konwersja etykiet
df['Label_num'] = df[label_col_name].map(label_mapping)
# Usuń wiersze z etykietami, których nie ma w mapowaniu (teraz usunie Trombocytopenię, jeśli została wygenerowana w poprzednim kroku)
df = df.dropna(subset=['Label_num'])
df['Label_num'] = df['Label_num'].astype(int)


X = df[features] # X ma teraz dane dla 25 cech
y = df['Label_num'] # Y ma teraz tylko 6 klas numerycznych (0-5)

print(f"Liczba próbek po mapowaniu etykiet: {len(df)}") # Będzie mniej próbek, bo usunięto Trombocytopenię
print(f"Liczba cech (parametrów z kartki): {len(features)}") # Powinno być 25
print(f"Nazwy cech: {features}")
print(f"Liczba unikalnych klas w całym zbiorze: {len(np.unique(y))}") # Powinno być 6
print(f"Unikalne etykiety numeryczne w całym zbiorze: {sorted(np.unique(y))}") # Powinno być [0, 1, 2, 3, 4, 5]
print(f"Mapowanie etykiet: {label_mapping}")


# === Podział danych na treningowy i testowy ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify=y zadba o rozkład 6 klas
)

print(f"\nRozmiar zbioru treningowego: {len(X_train)}")
print(f"Rozmiar zbioru testowego: {len(X_test)}")
print("\nRozkład klas w zbiorze treningowym:")
print(y_train.value_counts().sort_index())
print("\nRozkład klas w zbiorze testowym:")
print(y_test.value_counts().sort_index())

unique_y_train = sorted(np.unique(y_train))
print(f"\nUnikalne klasy numeryczne w zbiorze treningowym: {unique_y_train}") # Powinno być [0, 1, 2, 3, 4, 5]
if not all(cls in FULL_NUMERICAL_CLASSES for cls in unique_y_train):
    print("Warning: Klasy w y_train nie są zgodne z pełnym zestawem klas numerycznych!")


# === Scaling (Bez PCA) ===
print("\nSkalowanie danych (Bez PCA)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Skalowanie 25 cech
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(preprocess_dir, f"{script_name}_scaler.pkl"))

# === Wytrenowanie pomocniczego modelu XGBoost TYLKO DLA FEATURE IMPORTANCE ===
print("\nTrening pomocniczego XGBoost dla Feature Importance...")
xgb_importance_model = XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    objective='multi:softprob',
    num_class=len(FULL_NUMERICAL_CLASSES) # num_class = 6
)

# Trenujemy na skalowanych danych TRENINGOWYCH (25 cech)
# Jawnie podajemy pełny zestaw klas [0, 1, 2, 3, 4, 5]
xgb_importance_model.fit(X_train_scaled, y_train)
print("Pomocniczy XGBoost wytrenowany.")

# === Analiza Feature Importance ===
print("\nAnaliza ważności cech z pomocniczego XGBoost:")
feature_importances = xgb_importance_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features, # Lista 'features' ma 25 nazw
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
importance_df.to_csv(f"{script_name}_feature_importance_anemia_only.csv", index=False) # Nowa nazwa pliku
print(f"\nWażność cech zapisana do {script_name}_feature_importance_anemia_only.csv")


# === Train XGBoost (główny model) ===
print("\nTrening głównego XGBoost (na skalowanych danych)...")
xgb_model = XGBClassifier(eval_metric='mlogloss',
                          use_label_encoder=False,
                          random_state=42,
                          objective='multi:softprob',
                          num_class=len(FULL_NUMERICAL_CLASSES) # num_class = 6
                         )
# Trenujemy na SKALOWANYCH danych (Bez PCA)
# Jawnie podajemy pełny zestaw klas [0, 1, 2, 3, 4, 5]
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, f"{script_name}_xgb_anemia_only.pkl") # Nowa nazwa pliku
print("Główny XGBoost wytrenowany.")


# === Define PyTorch MLP ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim) # Output dim = liczba klas (6)
        )

    def forward(self, x):
        return self.model(x)

# === Train PyTorch MLP ===
print("\nTrening PyTorch MLP (na skalowanych danych)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Wejściowy wymiar MLP to liczba cech PO SKALOWANIU (25)
mlp_input_dim = X_train_scaled.shape[1] # Liczba cech po skalowaniu (25)
mlp_output_dim = len(FULL_NUMERICAL_CLASSES) # Liczba klas (6)
mlp_hidden_dim = max(64, mlp_input_dim * 2)

nn_model = MLP(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim).to(device)
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

X_test_tensor_eval = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor_eval = torch.tensor(y_test.values, dtype=torch.long).to(device)

train_dataset_mlp = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=128, shuffle=True)

num_epochs = 200

print("== Training NN ==")
for epoch in range(num_epochs):
    nn_model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_mlp:
        optimizer.zero_grad()
        output = nn_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    epoch_loss = train_loss / len(train_dataset_mlp)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        nn_model.eval()
        with torch.no_grad():
            val_output = nn_model(X_test_tensor_eval)
            val_loss = criterion(val_output, y_test_tensor_eval).item()
            val_pred = torch.argmax(val_output, dim=1)
            val_acc = (val_pred == y_test_tensor_eval).float().mean().item()
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={epoch_loss:.4f}")

torch.save(nn_model.state_dict(), f"{script_name}_mlp_anemia_only.pth") # Nowa nazwa pliku
print("MLP wytrenowany.")


# === Stacking z Logistic Regression (Meta-klasyfikator) ===
print("\nTrening meta-klasyfikatora (Stacking)...")

# 1. Uzyskaj prawdopodobieństwa z modeli bazowych na DANYCH TRENINGOWYCH (SKALOWANYCH)
xgb_proba_train = xgb_model.predict_proba(X_train_scaled)

nn_model.eval()
with torch.no_grad():
    X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    nn_proba_train = nn_model(X_train_scaled_tensor).softmax(dim=1).cpu().numpy()

# Sprawdzenie, czy obie macierze prawdopodobieństw mają tę samą liczbę kolumn (6)
if xgb_proba_train.shape[1] != nn_proba_train.shape[1]:
     raise ValueError(f"Mismatch in number of classes predicted by base models on train data. XGBoost: {xgb_proba_train.shape[1]}, MLP: {nn_proba_train.shape[1]}")

stacked_input_train = np.hstack([xgb_proba_train, nn_proba_train])

stacking_model = LogisticRegression(max_iter=1000, random_state=42)
# Trenujemy na 6 klasach numerycznych [0, 1, 2, 3, 4, 5]
stacking_model.fit(stacked_input_train, y_train)

joblib.dump(stacking_model, f"{script_name}_stacking_model_anemia_only.pkl") # Nowa nazwa pliku
print("Meta-klasyfikator wytrenowany.")


# === Ewaluacja całego stosu (Meta-klasyfikatora) ===
print("\nEwaluacja meta-klasyfikatora na danych testowych...")

# 1. Uzyskaj prawdopodobieństwa z modeli bazowych na DANYCH TESTOWYCH (SKALOWANYCH)
xgb_proba_test = xgb_model.predict_proba(X_test_scaled)

nn_model.eval()
with torch.no_grad():
    X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    nn_proba_test = nn_model(X_test_scaled_tensor).softmax(dim=1).cpu().numpy()

# Sprawdzenie zgodności liczby kolumn na zbiorze testowym (6)
if xgb_proba_test.shape[1] != nn_proba_test.shape[1]:
     raise ValueError(f"Mismatch in number of classes predicted by base models on test data. XGBoost: {xgb_proba_test.shape[1]}, MLP: {nn_proba_test.shape[1]}")

stacked_input_test = np.hstack([xgb_proba_test, nn_proba_test])

y_pred_final = stacking_model.predict(stacked_input_test)
y_score = stacking_model.predict_proba(stacked_input_test)


# === Obliczanie metryk ===
acc = accuracy_score(y_test, y_pred_final)
print(f"Accuracy (meta-klasyfikator): {acc:.4f}")

# === Przygotowanie danych i etykiet do wykresów ===

# stacking_model.classes_ to numeryczne etykiety klas, których model się nauczył.
# Po usunięciu Trombocytopenii, powinno to być [0, 1, 2, 3, 4, 5].
model_learned_classes = stacking_model.classes_
n_model_classes = len(model_learned_classes) # Będzie 6

# Binarizujemy y_test względem 6 klas [0, 1, 2, 3, 4, 5]
y_test_bin = label_binarize(y_test, classes=model_learned_classes)

# class_names_display - lista nazw klas do wyświetlenia.
# Upewnij się, że pasują do numerycznych etykiet w model_learned_classes.
num_to_label = {num: name for name, num in label_mapping.items()}
class_names_display = [num_to_label[num] for num in model_learned_classes] # Będzie ['Anemia Mikrocytarna', ..., 'Healthy']


print(f"\nKlasy, których nauczył się model (numeryczne): {model_learned_classes}") # Powinno być [0, 1, 2, 3, 4, 5]
print(f"Nazwy klas do wyświetlenia (w kolejności modelu): {class_names_display}") # Powinno być 6 nazw
print(f"Kształt y_test_bin: {y_test_bin.shape}") # Powinno być (liczba próbek testowych, 6)
print(f"Kształt y_score: {y_score.shape}") # Powinno być (liczba próbek testowych, 6)


# === ROC Curve ===
print("Generowanie krzywej ROC...")
fpr, tpr, roc_auc = {}, {}, {}
# Iterujemy przez 6 klas
for i in range(n_model_classes):
    class_num = model_learned_classes[i]
    fpr[class_num], tpr[class_num], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[class_num] = auc(fpr[class_num], tpr[class_num])


plt.figure(figsize=(10, 8))
colors = plt.get_cmap('tab10')
# Iterujemy przez 6 klas
for i in range(n_model_classes):
     class_num = model_learned_classes[i]
     class_display_name = class_names_display[i]
     plt.plot(fpr[class_num], tpr[class_num], lw=2, color=colors(i % colors.N),
              label=f'{class_display_name} (AUC = {roc_auc[class_num]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - One vs Rest (meta-klasyfikator)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{script_name}_roc_meta_classifier_anemia_only.png") # Nowa nazwa pliku
print(f"Krzyba ROC zapisana jako {script_name}_roc_meta_classifier_anemia_only.png")

# === Confusion Matrix ===
print("Generowanie macierzy pomyłek...")
# Macierz pomyłek dla 6 klas
cm = confusion_matrix(y_test, y_pred_final, labels=model_learned_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_display)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (meta-klasyfikator)")
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{script_name}_confusion_matrix_meta_anemia_only.png") # Nowa nazwa pliku
print(f"Macierz pomyłek zapisana jako {script_name}_confusion_matrix_meta_anemia_only.png")

plt.show()

# === Zapis metadanych (dostosowane do 6 klas, bez PCA) ===
import json

num_to_label_map = {num: name for name, num in label_mapping.items()}
model_classes_in_order = stacking_model.classes_ # Będzie [0, 1, 2, 3, 4, 5]
model_class_names_in_order = [num_to_label_map[num] for num in model_classes_in_order] # Będzie 6 nazw

metadata = {
    "label_mapping": label_mapping, # Mapowanie dla 6 klas
    "full_numerical_classes": FULL_NUMERICAL_CLASSES, # [0, 1, 2, 3, 4, 5]
    "model_classes_order": model_classes_in_order.tolist(),
    "model_class_names_order": model_class_names_in_order,
    "feature_names": features, # 25 cech z kartki
    "used_pca": False,
    "num_features_after_preprocessing": X_train_scaled.shape[1] # 25 cech
}

with open(f"{script_name}_metadata_anemia_only.json", "w") as f: # Nowa nazwa pliku
    json.dump(metadata, f)

print("\nArtefakty modelu i metadane (anemia only, bez PCA) zapisane.")