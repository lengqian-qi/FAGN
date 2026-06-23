import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# 1. Frenet-aware Graph Conv
# ===============================
class FrenetAwareGraphConv(nn.Module):
    def __init__(self, in_features, out_features, k=6):
        super().__init__()
        self.k = k

        self.mlp = nn.Sequential(
            nn.Linear(in_features + 3, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, points, T, N, B):
        """
        x: (B, P, C)
        points: (B, P, 3)
        """

        B_size, P, C = x.shape
        outputs = []

        for b in range(B_size):

            # ---------- KNN ----------
            dist = torch.cdist(points[b], points[b])  # (P, P)
            knn = dist.topk(k=self.k + 1, largest=False)[1][:, 1:]  # (P, k)

            # ---------- neighbor features ----------
            neighbor_feat = x[b][knn]  # (P, k, C)

            # ---------- geometry ----------
            center_points = points[b].unsqueeze(1)   # (P,1,3)
            neighbor_points = points[b][knn]         # (P,k,3)

            relative = neighbor_points - center_points  # (P,k,3)

            # ---------- Frenet frame ----------
            T_center = T[b].unsqueeze(1).expand(-1, self.k, -1)
            N_center = N[b].unsqueeze(1).expand(-1, self.k, -1)
            B_center = B[b].unsqueeze(1).expand(-1, self.k, -1)

            R = torch.stack([T_center, N_center, B_center], dim=2)  # (P,k,3,3)

            # ---------- local transform ----------
            local_pos = torch.matmul(
                R.transpose(2, 3),
                relative.unsqueeze(-1)
            ).squeeze(-1)  # (P,k,3)

            # ---------- concat ----------
            combined = torch.cat([neighbor_feat, local_pos], dim=-1)  # (P,k,C+3)

            combined = combined.reshape(-1, C + 3)
            transformed = self.mlp(combined)
            transformed = transformed.reshape(P, self.k, -1)

            # ---------- pooling ----------
            agg = torch.max(transformed, dim=1)[0]  # (P,out)

            outputs.append(agg)

        return torch.stack(outputs)  # (B,P,out)


# ===============================
# 2. Dual-branch PointNet
# ===============================
class EnhancedPointNetfeat(nn.Module):
    def __init__(self, use_frenet=True):
        super().__init__()
        self.use_frenet = use_frenet

        # -------- PointNet backbone --------
        self.conv1 = nn.Conv1d(12, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # -------- Frenet branch --------
        if use_frenet:
            self.frenet_conv = FrenetAwareGraphConv(128, 128)

            # learnable weight
            self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """
        x: (B, 12, P)
        """

        B, _, P = x.shape

        coords = x[:, :3, :]            # (B,3,P)
        T = x[:, 3:6, :].transpose(1, 2)
        N = x[:, 6:9, :].transpose(1, 2)
        B_vec = x[:, 9:12, :].transpose(1, 2)

        # -------- PointNet --------
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))   # (B,128,P)

        # -------- Frenet--------
        if self.use_frenet:
            x_frenet = x.transpose(1, 2)  # (B,P,128)

            x_frenet = self.frenet_conv(
                x_frenet,
                coords.transpose(1, 2),
                T, N, B_vec
            )

            x_frenet = x_frenet.transpose(1, 2)  # (B,128,P)

            # FUSION
            x = x + self.alpha * x_frenet

        # -------- Global --------
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]

        return x


# ===============================
# 3. Classifier
# ===============================
class FrenetPointNetCls(nn.Module):
    def __init__(self, k=72, use_frenet=True):
        super().__init__()

        self.feat = EnhancedPointNetfeat(use_frenet)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)