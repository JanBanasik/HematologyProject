# === generate.py (Modyfikacja - Bardziej Agresywne Nakładanie/Szum) ===
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ustawienia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 128
epochs = 200 # Zwiększamy epoki VAE, bo dane są trudniejsze
learning_rate = 1e-3
latent_dim = 25

# --- LISTA PARAMETRÓW Z KARTKI (BEZ ZMIAN) ---
CARD_PARAMETERS = [
    'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
    'RDW-SD', 'RDW-CV',
    'PDW', 'MPV', 'PLCR', 'PCT',
    'NEUT#', 'LYMPH#', 'MONO#', 'EOS#', 'BASO#',
    'NRBC#',
    'RET', 'IRF', 'LFR', 'MFR', 'HFR'
]

# --- SZACOWANE ZAKRESY REFERENCYJNE I ŚREDNIE/STD DLA STANU 'N' ---
# Lekko zwiększamy STD dla niektórych parametrów, żeby rozkłady "normalne" były szersze
NORMAL_PROFILES_BASE = {
    'WBC': {'mean': 7.0, 'std': 2.5, 'range': (4.0, 10.0)}, # Zwiększono std
    'NEUT#': {'mean': 4.5, 'std': 1.8, 'range': (2.0, 7.0)}, # Zwiększono std
    'LYMPH#': {'mean': 2.0, 'std': 1.0, 'range': (1.0, 3.0)}, # Zwiększono std
    'MONO#': {'mean': 0.5, 'std': 0.4, 'range': (0.2, 1.0)},
    'EOS#': {'mean': 0.15, 'std': 0.12, 'range': (0.05, 0.5)},
    'BASO#': {'mean': 0.03, 'std': 0.03, 'range': (0.01, 0.1)},
    'RBC': {'mean': 5.0, 'std': 0.6, 'range': (4.0, 6.0)}, # Zwiększono std
    'HGB': {'mean': 14.5, 'std': 1.8, 'range': (12.0, 17.0)}, # Zwiększono std
    'HCT': {'mean': 43.0, 'std': 4.0, 'range': (36.0, 50.0)}, # Zwiększono std
    'MCV': {'mean': 90.0, 'std': 6.0, 'range': (80.0, 100.0)}, # Zwiększono std
    'MCH': {'mean': 29.0, 'std': 2.5, 'range': (27.0, 33.0)}, # Zwiększono std
    'MCHC': {'mean': 34.0, 'std': 1.2, 'range': (33.0, 36.0)},
    'RDW-SD': {'mean': 42.0, 'std': 4.0, 'range': (35.0, 50.0)}, # Zwiększono std
    'RDW-CV': {'mean': 13.0, 'std': 1.5, 'range': (11.5, 14.5)}, # Zwiększono std
    'NRBC#': {'mean': 0.01, 'std': 0.03, 'range': (0.0, 0.05)}, # Zwiększono std
    'PLT': {'mean': 280, 'std': 100, 'range': (150, 450)}, # Zwiększono std
    'PDW': {'mean': 12.0, 'std': 2.5, 'range': (9.0, 15.0)},
    'MPV': {'mean': 9.5, 'std': 1.2, 'range': (7.5, 11.0)},
    'PLCR': {'mean': 25.0, 'std': 6.0, 'range': (15.0, 35.0)},
    'PCT': {'mean': 0.25, 'std': 0.07, 'range': (0.18, 0.35)},

    # === Dodane parametry retikulocytów ===
    'RET': {'mean': 1.0, 'std': 0.7, 'range': (0.5, 2.0)}, # Zwiększono std
    'IRF': {'mean': 15.0, 'std': 7.0, 'range': (10.0, 25.0)}, # Zwiększono std
    'LFR': {'mean': 80.0, 'std': 12.0, 'range': (60.0, 90.0)},
    'MFR': {'mean': 15.0, 'std': 7.0, 'range': (5.0, 25.0)}, # Zwiększono std
    'HFR': {'mean': 5.0, 'std': 4.0, 'range': (0.0, 15.0)}, # Zwiększono std
}

# --- DEFINICJA PROFILI DLA POSZCZEGÓLNYCH KLAS (BEZ ZMIAN) ---
GROUP_PROFILES_SHIFTS = {
    "Healthy": {param: 'N' for param in CARD_PARAMETERS},

    "Anemia Mikrocytarna": {
        'RBC': '↓ or N', 'HGB': '↓', 'HCT': '↓', 'MCV': '↓', 'MCH': '↓', 'MCHC': '↓',
        'RDW-SD': '↑ or N', 'RDW-CV': '↑ or N',
        'PLT': '↑ or N',
        'RET': 'N', 'IRF': 'N', 'LFR': 'N', 'MFR': 'N', 'HFR': 'N',
        'NRBC#': 'N',
    },

    "Anemia Makrocytarna": {
        'RBC': '↓', 'HGB': '↓', 'HCT': '↓', 'MCV': '↑', 'MCH': '↑', 'MCHC': 'N',
        'RDW-SD': '↑', 'RDW-CV': '↑',
        'PLT': '↓ or N',
        'NRBC#': '↑ or N',
        'NEUT#': '↓ or N',
        'RET': 'N or ↑', 'IRF': 'N or ↑', 'LFR': 'N', 'MFR': '↑', 'HFR': '↑',
    },

    "Anemia Normocytarna": {
        'RBC': '↓', 'HGB': '↓', 'HCT': '↓', 'MCV': 'N', 'MCH': 'N', 'MCHC': 'N',
        'RDW-SD': 'N or ↑', 'RDW-CV': 'N or ↑',
        'RET': 'N or ↓', 'IRF': 'N or ↓',
        'NRBC#': 'N',
    },
     "Anemia Hemolityczna": {
        'RBC': '↓', 'HGB': '↓', 'HCT': '↓', 'MCV': 'N', 'MCH': 'N', 'MCHC': 'N',
        'RDW-SD': '↑', 'RDW-CV': '↑',
        'NRBC#': '↑',
        'PLT': '↑ or N',
        'WBC': '↑ or N', 'NEUT#': '↑ or N',
        'RET': '↑', 'IRF': '↑', 'LFR': 'N', 'MFR': '↑', 'HFR': '↑',
    },

    "Anemia Aplastyczna": {
        'RBC': '↓', 'HGB': '↓', 'HCT': '↓', 'MCV': 'N or ↑', 'MCH': 'N', 'MCHC': 'N',
        'RDW-SD': 'N or ↑', 'RDW-CV': 'N or ↑',
        'PLT': '↓',
        'WBC': '↓', 'NEUT#': '↓', 'LYMPH#': '↓', 'MONO#': '↓', 'EOS#': '↓', 'BASO#': '↓',
        'RET': '↓', 'IRF': '↓', 'LFR': '↓', 'MFR': '↓', 'HFR': '↓',
        'NRBC#': '↓ or N',
    },
     # Możesz dodać profile dla Infekcji/Krwotoku, jeśli chcesz je modelować
     # "Infekcja Wirusowa": {...}
     # "Infekcja Bakteryjna": {...}
     # "Krwotok Ostry": {...}
     # "Krwotok Przewlekły": {...}
}


def generate_value_realistic(param_name, shift_type, n_samples, overlap_factor=0.4, noise_factor=0.3):
    base_info = NORMAL_PROFILES_BASE.get(param_name)
    if not base_info:
        print(f"WARNING: Brak danych bazowych dla parametru: {param_name}. Generowanie losowe z zakresu 0-1.")
        return np.random.rand(n_samples)  # Fallback

    base_mean = base_info['mean']
    base_std = base_info['std']
    param_range = base_info['range']

    low_mean = base_mean - base_std * (2.0 - 1.5 * overlap_factor)
    high_mean = base_mean + base_std * (2.0 - 1.5 * overlap_factor)
    shifted_std = max(base_std * (1.0 + 0.8 * overlap_factor), 1e-6)
    base_std = max(base_std, 1e-6)

    def safe_split(n, ratios):
        """
        Dzieli n na części proporcjonalnie do ratios tak, aby suma się zgadzała i wszystkie wyniki były int.
        """
        raw = [r / sum(ratios) * n for r in ratios]
        rounded = [int(x) for x in raw]
        diff = n - sum(rounded)
        for i in range(abs(diff)):
            rounded[i % len(rounded)] += 1 if diff > 0 else -1
        return rounded

    if shift_type == 'N':
        values = np.random.normal(base_mean, base_std, n_samples)

    elif shift_type == '↓':
        values = np.random.normal(low_mean, shifted_std, n_samples)

    elif shift_type == '↑':
        values = np.random.normal(high_mean, shifted_std, n_samples)

    elif shift_type == '↓ or N':
        prob_low = min(0.6 + overlap_factor / 2, 0.9)
        weights = [prob_low, 1.0 - prob_low]
        n_low, n_norm = safe_split(n_samples, weights)
        print(prob_low, weights, n_low, n_norm)
        values = np.concatenate([
            np.random.normal(low_mean, shifted_std, n_low),
            np.random.normal(base_mean, base_std, n_norm)
        ])
        np.random.shuffle(values)

    elif shift_type == '↑ or N':
        prob_high = min(0.6 + overlap_factor / 2, 0.9)
        weights = [prob_high, 1.0 - prob_high]
        n_high, n_norm = safe_split(n_samples, weights)
        values = np.concatenate([
            np.random.normal(high_mean, shifted_std, n_high),
            np.random.normal(base_mean, base_std, n_norm)
        ])
        np.random.shuffle(values)

    elif shift_type == '↓ or ↑ or N':
        weights = [1/3, 1/3, 1/3]
        n_low, n_high, n_norm = safe_split(n_samples, weights)
        values = np.concatenate([
            np.random.normal(low_mean, shifted_std, n_low),
            np.random.normal(high_mean, shifted_std, n_high),
            np.random.normal(base_mean, base_std, n_norm)
        ])
        np.random.shuffle(values)

    else:
        print(f"WARNING: Nieznany shift_type '{shift_type}' dla parametru {param_name}. Przyjęto 'N'.")
        values = np.random.normal(base_mean, base_std, n_samples)

    # Dodaj szum
    values += np.random.normal(0, base_std * noise_factor, n_samples)

    # Ograniczenie wartości
    if param_range:
        buffer = (param_range[1] - param_range[0]) * 1.0
        min_limit = param_range[0] - buffer
        max_limit = param_range[1] + buffer
        params_geq_0 = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'RDW-SD', 'PDW', 'MPV', 'PCT', 'NEUT#', 'LYMPH#', 'MONO#', 'EOS#', 'BASO#', 'NRBC#', 'RET', 'IRF', 'LFR', 'MFR', 'HFR', 'RDW-CV', 'PLCR']
        if param_name in params_geq_0:
            min_limit = max(0.0, min_limit)
        values = np.clip(values, min_limit, max_limit)

    return values



# --- Funkcja do generowania danych dla pojedynczej grupy ---
def generate_realistic_group_data(label, n_samples=random.randint(4000, 6000)):
    data = {}
    profile_shifts = GROUP_PROFILES_SHIFTS.get(label)
    if not profile_shifts:
        raise ValueError(f"Nie zdefiniowano profilu przesunięć dla etykiety: {label}")

    # BARDZIEJ AGRESYWNE ZAKRESY DLA NAKŁADANIA I SZUMU
    group_overlap_factor = random.uniform(0.7, 1.0) # Zwiększono zakres
    group_noise_factor = random.uniform(0.4, 0.7) # Zwiększono zakres

    for param in CARD_PARAMETERS:
        shift_type = profile_shifts.get(param, 'N')

        data[param] = generate_value_realistic(
            param, shift_type, n_samples,
            overlap_factor=group_overlap_factor,
            noise_factor=group_noise_factor
        )

    df = pd.DataFrame(data)
    df['Label'] = label

    # --- Opcjonalnie: Dodaj losowe "outliers" (BEZ ZMIAN) ---
    num_outliers = int(n_samples * random.uniform(0.005, 0.01))
    if num_outliers > 0:
         outlier_indices = np.random.choice(n_samples, num_outliers, replace=False)
         outlier_params_to_change = np.random.choice(CARD_PARAMETERS, size=int(num_outliers * 1.5), replace=True)
         for i in outlier_indices:
             params_for_this_outlier = np.random.choice(outlier_params_to_change, size=random.randint(1, 5), replace=False)
             for param in params_for_this_outlier:
                 base_info = NORMAL_PROFILES_BASE.get(param)
                 if base_info and base_info['range']:
                     min_val, max_val = base_info['range']
                     buffer_outlier = (max_val - min_val) * random.uniform(1.5, 4.0)
                     extreme_val = np.random.uniform(min_val - buffer_outlier, max_val + buffer_outlier)
                     params_geq_0 = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'RDW-SD', 'PDW', 'MPV', 'PCT', 'NEUT#', 'LYMPH#', 'MONO#', 'EOS#', 'BASO#', 'NRBC#', 'RET', 'IRF', 'LFR', 'MFR', 'HFR', 'RDW-CV', 'PLCR']
                     if param in params_geq_0:
                         extreme_val = max(0.0, extreme_val)
                     df.loc[i, param] = extreme_val
                 else:
                     df.loc[i, param] = np.random.uniform(-200, 200)

    return df

# --- Generowanie całego zbioru danych (BEZ ZMIAN - już usunięto Trombocytopenię) ---
LABELS_TO_GENERATE = [
    "Healthy",
    "Anemia Mikrocytarna",
    "Anemia Makrocytarna",
    "Anemia Normocytarna",
    "Anemia Hemolityczna",
    "Anemia Aplastyczna",
    # Dodaj inne klasy, jeśli zdefiniowałeś dla nich profile
]

all_data_frames = []
total_samples = 0
for label in LABELS_TO_GENERATE:
    n_samples = random.randint(4000, 6000)
    print(f"Generowanie {n_samples} próbek dla: {label}")
    all_data_frames.append(generate_realistic_group_data(label, n_samples=n_samples))
    total_samples += n_samples

df_realistic = pd.concat(all_data_frames, ignore_index=True)
df_realistic = df_realistic.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nWygenerowano {total_samples} realistycznych próbek.")
print("Przykładowe dane realistyczne (pierwsze 5 wierszy i pierwsze 10 kolumn):")
print(df_realistic.head().iloc[:, :10])
print("\nPełna lista kolumn (cech):")
print(df_realistic.columns.tolist()) # Powinno być 25 cech
print("\nRozkład klas:")
print(df_realistic['Label'].value_counts())

df_realistic.to_csv("synthetic_data_card_parameters_anemia_only.csv", index=False)
print("\nWygenerowane dane zapisane do: synthetic_data_card_parameters_anemia_only.csv")

# --- Przygotowanie danych do VAE i klasyfikatorów (logika bez zmian, operuje na nowych danych) ---
# Identyfikacja kolumn
numeric_cols_realistic = [col for col in df_realistic.columns if col != 'Label']
label_col_name = 'Label'

# Kodowanie etykiet (one-hot encoding)
df_encoded_realistic = pd.get_dummies(df_realistic, columns=[label_col_name], prefix='Label')

label_cols_realistic = [col for col in df_encoded_realistic.columns if col.startswith('Label_')]
numeric_cols_realistic.sort()
label_cols_realistic.sort()
all_cols_ordered = numeric_cols_realistic + label_cols_realistic
df_encoded_realistic = df_encoded_realistic[all_cols_ordered]

label_start_idx_realistic = df_encoded_realistic.columns.get_loc(label_cols_realistic[0])
input_dim_realistic = df_encoded_realistic.shape[1] # Liczba cech (25) + liczba klas (6) = 31

print(f"\nNowe kolumny liczbowe ({len(numeric_cols_realistic)}): {numeric_cols_realistic}")
print(f"Nowe kolumny etykiet ({len(label_cols_realistic)}): {label_cols_realistic}")
print(f"Kolumny etykiet zaczynają się od indeksu: {label_start_idx_realistic}")
print(f"Nowe input_dim dla VAE: {input_dim_realistic}")
print(f"Kształt zakodowanego DataFrame: {df_encoded_realistic.shape}")


# Normalizacja
mean_realistic = df_encoded_realistic[numeric_cols_realistic].mean()
std_realistic = df_encoded_realistic[numeric_cols_realistic].std()

df_encoded_realistic[numeric_cols_realistic] = (df_encoded_realistic[numeric_cols_realistic] - mean_realistic) / std_realistic
df_encoded_realistic = df_encoded_realistic.astype('float32')

# Tensory i DataLoader
data_tensor_realistic = torch.tensor(df_encoded_realistic.values, dtype=torch.float32)
dataset_realistic = TensorDataset(data_tensor_realistic)
dataloader_realistic = DataLoader(dataset_realistic, batch_size=batch_size, shuffle=True)

# --- VAE Model Definition (BEZ ZMIAN) ---
class VAE_Realistic(nn.Module):
    def __init__(self, input_dim, latent_dim, label_start_idx):
        super(VAE_Realistic, self).__init__()
        self.label_start_idx = label_start_idx
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, latent_dim)
        self.fc32 = nn.Linear(64, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        output = self.fc6(h5)
        output[:, self.label_start_idx:] = torch.sigmoid(output[:, self.label_start_idx:])
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Loss function for VAE (BEZ ZMIAN)
def loss_function_vae(recon_x, x, mu, logvar, label_start_idx):
    mse_loss = nn.MSELoss(reduction='sum')(recon_x[:, :label_start_idx], x[:, :label_start_idx])
    bce_loss = nn.BCELoss(reduction='sum')(recon_x[:, label_start_idx:], x[:, label_start_idx:])
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + bce_loss + KLD

model_realistic = VAE_Realistic(input_dim=input_dim_realistic, latent_dim=latent_dim, label_start_idx=label_start_idx_realistic).to(device)
optimizer_realistic = optim.Adam(model_realistic.parameters(), lr=learning_rate)

# --- Training the Realistic VAE Model ---
print("== Training Realistic VAE ==")
model_realistic.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch in dataloader_realistic:
        data = batch[0].to(device)
        optimizer_realistic.zero_grad()
        recon_batch, mu, logvar = model_realistic(data)
        loss = loss_function_vae(recon_batch, data, mu, logvar, label_start_idx_realistic)
        loss.backward()
        train_loss += loss.item()
        optimizer_realistic.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {train_loss / len(dataset_realistic):.4f}")

# --- Generating Synthetic Data from the Trained VAE ---
model_realistic.eval()
with torch.no_grad():
    z = torch.randn(len(dataset_realistic), latent_dim).to(device)
    generated_realistic_raw = model_realistic.decode(z).cpu().numpy()

# --- Post-processing Generated Data ---
df_generated_realistic = pd.DataFrame(generated_realistic_raw, columns=df_encoded_realistic.columns)

# 1. Reverse Normalization
for col in numeric_cols_realistic:
    if std_realistic[col] != 0:
       df_generated_realistic[col] = df_generated_realistic[col] * std_realistic[col] + mean_realistic[col]
    else:
       df_generated_realistic[col] = mean_realistic[col]

# 2. Convert One-Hot Encoded Labels back to Class Names
generated_labels_encoded = df_generated_realistic[label_cols_realistic]
predicted_label_indices = np.argmax(generated_labels_encoded.values, axis=1)
index_to_label_name = {i: col.replace("Label_", "") for i, col in enumerate(label_cols_realistic)}
generated_labels = [index_to_label_name[idx] for idx in predicted_label_indices]

# 3. Create the Final Generated DataFrame
df_generated_final = df_generated_realistic[numeric_cols_realistic].copy()
df_generated_final['Label'] = generated_labels

print("\nPrzykładowe wygenerowane dane z realistycznego VAE (pierwsze 5 wierszy i pierwsze 10 kolumn):")
print(df_generated_final.head().iloc[:, :10])
print("\nRozkład klas w wygenerowanych danych:")
print(df_generated_final['Label'].value_counts())

df_generated_final.to_csv("synthetic_data_card_parameters_anemia_only_generated_vae.csv", index=False)
print("\nWygenerowane dane z VAE zapisane do: synthetic_data_card_parameters_anemia_only_generated_vae.csv")