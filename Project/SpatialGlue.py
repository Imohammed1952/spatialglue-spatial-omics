import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class IntraModalityAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.randn(dim))

    def score(self, h):
        u = torch.tanh(self.W(h))
        s = torch.matmul(u, self.q)
        return s

    def forward(self, h_spatial, h_feature):
        s1 = self.score(h_spatial)
        s2 = self.score(h_feature)

        scores = torch.stack([s1, s2], dim=1)   # (N, 2)
        alpha = torch.softmax(scores, dim=1)    # (N, 2)

        out = alpha[:, 0:1] * h_spatial + alpha[:, 1:2] * h_feature
        return out, alpha


class InterModalityAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.randn(dim))

    def score(self, h):
        u = torch.tanh(self.W(h))
        s = torch.matmul(u, self.q)
        return s

    def forward(self, modality_embeddings):
        scores = [self.score(h) for h in modality_embeddings]
        scores = torch.stack(scores, dim=1)   # (N, M)

        beta = torch.softmax(scores, dim=1)   # (N, M)

        z = torch.zeros_like(modality_embeddings[0])
        for m, h in enumerate(modality_embeddings):
            z = z + beta[:, m:m+1] * h

        return z, beta


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class SpatialGlueMini(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_modalities=2, dropout=0.0):
        super().__init__()

        self.num_modalities = num_modalities

        self.spatial_encoders = nn.ModuleList([
            GCNEncoder(in_dim, hidden_dim, latent_dim, dropout)
            for _ in range(num_modalities)
        ])

        self.feature_encoders = nn.ModuleList([
            GCNEncoder(in_dim, hidden_dim, latent_dim, dropout)
            for _ in range(num_modalities)
        ])

        self.intra_attentions = nn.ModuleList([
            IntraModalityAttention(latent_dim)
            for _ in range(num_modalities)
        ])

        self.inter_attention = InterModalityAttention(latent_dim)

        self.decoders = nn.ModuleList([
            Decoder(latent_dim, hidden_dim, in_dim)
            for _ in range(num_modalities)
        ])

    def forward(self, xs, spatial_edge_index, feature_edge_indices):
        modality_embeddings = []
        intra_alphas = []

        for m in range(self.num_modalities):
            h_spatial = self.spatial_encoders[m](xs[m], spatial_edge_index)
            h_feature = self.feature_encoders[m](xs[m], feature_edge_indices[m])

            y_m, alpha_m = self.intra_attentions[m](h_spatial, h_feature)
            modality_embeddings.append(y_m)
            intra_alphas.append(alpha_m)

        z, beta = self.inter_attention(modality_embeddings)

        recons = [self.decoders[m](z) for m in range(self.num_modalities)]

        return {
            "z": z,
            "modality_embeddings": modality_embeddings,
            "recons": recons,
            "intra_alphas": intra_alphas,
            "inter_betas": beta
        }


def reconstruction_loss(xs, recons):
    loss = torch.tensor(0.0, device=xs[0].device)
    for x, x_hat in zip(xs, recons):
        loss = loss + F.mse_loss(x_hat, x)
    return loss


def correspondence_loss(modality_embeddings):
    loss = torch.tensor(0.0, device=modality_embeddings[0].device)
    count = 0

    for i in range(len(modality_embeddings)):
        for j in range(i + 1, len(modality_embeddings)):
            loss = loss + F.mse_loss(modality_embeddings[i], modality_embeddings[j])
            count += 1

    return loss / max(count, 1)


def total_loss(xs, outputs, lambda_corr=1.0):
    recon = reconstruction_loss(xs, outputs["recons"])
    corr = correspondence_loss(outputs["modality_embeddings"])
    total = recon + lambda_corr * corr
    return total, recon, corr


def train_model(
    model,
    xs,
    spatial_edge_index,
    feature_edge_indices,
    epochs=500,
    lr=1e-4,
    lambda_corr=1.0,
    device="cpu"
):
    model = model.to(device)

    xs = [x.to(device) for x in xs]
    spatial_edge_index = spatial_edge_index.to(device)
    feature_edge_indices = [e.to(device) for e in feature_edge_indices]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(xs, spatial_edge_index, feature_edge_indices)
        loss, recon, corr = total_loss(xs, outputs, lambda_corr)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"total={loss.item():.4f} | "
                f"recon={recon.item():.4f} | "
                f"corr={corr.item():.4f}"
            )

    return model