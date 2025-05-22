#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CVAE(nn.Module):
    def __init__(self, input_dim, label_dim, latent_dim=10, hidden_dim=64):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + label_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim + label_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x, y):
        h1 = self.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = self.relu(self.fc3(torch.cat([z, y], dim=1)))
        return self.fc4(h3)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss.item(), kld.item()


def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, total_recon, total_kld = 0, 0, 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            loss, recon, kld = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            total_recon += recon
            total_kld += kld
            optimizer.step()
        print(f"Epoch {epoch}: Total={total_loss:.2f}, Recon={total_recon:.2f}, KLD={total_kld:.2f}")


def generate_samples(model, scaler, le, n_samples, latent_dim):
    model.eval()
    features = ['HCT', 'HGB', 'MCH', 'MCHC', 'MCV', 'PLT', 'RBC', 'RDW', 'WBC']
    classes = le.classes_
    n_classes = len(classes)
    base = n_samples // n_classes
    extras = n_samples % n_classes
    samples_per_class = [base + (1 if i < extras else 0) for i in range(n_classes)]

    synthetic = []
    with torch.no_grad():
        for i, count in enumerate(samples_per_class):
            y = torch.zeros(count, n_classes, device=device)
            y[:, i] = 1
            z = torch.randn(count, latent_dim, device=device)
            gen = model.decode(z, y).cpu().numpy()
            synthetic.append(gen)

    synt = np.vstack(synthetic)
    synt = scaler.inverse_transform(synt)
    labels = np.concatenate([[cls] * n for cls, n in zip(classes, samples_per_class)])
    df_syn = pd.DataFrame(synt, columns=features)
    df_syn['label'] = labels
    return df_syn


def visualize(real_df, synthetic_df, features, output_path_prefix):
    sns.set(style='whitegrid', palette='muted')

    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(real_df[feature], color='blue', label='Real', stat='density', bins=30, kde=True)
        sns.histplot(synthetic_df[feature], color='orange', label='Synthetic', stat='density', bins=30, kde=True)
        plt.title(f'Feature: {feature}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_hist_{feature}.png")
        plt.close()

    def plot_2D(data_real, data_synt, method, name):
        plt.figure(figsize=(6, 6))
        data = np.vstack([data_real, data_synt])
        labels = np.array(['Real'] * len(data_real) + ['Synthetic'] * len(data_synt))
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(data)
        df_plot = pd.DataFrame(reduced, columns=['Dim1', 'Dim2'])
        df_plot['Type'] = labels
        sns.scatterplot(data=df_plot, x='Dim1', y='Dim2', hue='Type', alpha=0.5)
        plt.title(f"{method.upper()} - Real vs Synthetic")
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_{method}.png")
        plt.close()

    real_scaled = real_df[features].values
    synth_scaled = synthetic_df[features].values
    plot_2D(real_scaled, synth_scaled, 'pca', output_path_prefix)
    try:
        plot_2D(real_scaled, synth_scaled, 'tsne', output_path_prefix)
    except:
        print("t-SNE visualization failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_samples', type=int, default=75000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=10)
    args = parser.parse_args()

    # Load and preprocess
    df = pd.read_csv(args.input)
    features = ['HCT', 'HGB', 'MCH', 'MCHC', 'MCV', 'PLT', 'RBC', 'RDW', 'WBC']
    df = df.dropna(subset=features, how='any').reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    Y = np.eye(len(le.classes_))[y]

    # DataLoader
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = CVAE(input_dim=len(features), label_dim=len(le.classes_), latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train(model, loader, optimizer, args.epochs)

    # Generate
    df_syn = generate_samples(model, scaler, le, args.n_samples, args.latent_dim)
    df_syn.to_csv(args.output, index=False)
    print(f"Synthetic data saved to {args.output}")

    # Visualize
    output_prefix = args.output.replace('.csv', '')
    visualize(df, df_syn, features, output_prefix)


if __name__ == '__main__':
    main()
