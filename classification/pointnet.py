# Minhyuk Sung (mhsung@kaist.ac.kr)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetCls(nn.Module):
    def __init__(self, in_dim=3, n_classes=2):
        """
        PointNet: Deep Learning on Point Sets for 3D Classification and
        Segmentation.
        Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas.
        """
        super(PointNetCls, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes

        self.mlp_dims = [1024, 512, 256, self.n_classes]
        self.n_mlp_layers = len(self.mlp_dims) - 1
        assert(self.n_mlp_layers >= 1)

        self.dropout_prop = 0.5

        self.conv1 = nn.Conv1d(self.in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.n1 = nn.BatchNorm1d(64)
        self.n2 = nn.BatchNorm1d(64)
        self.n3 = nn.BatchNorm1d(64)
        self.n4 = nn.BatchNorm1d(128)
        self.n5 = nn.BatchNorm1d(1024)

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.do = nn.ModuleList()

        for i in range(self.n_mlp_layers):
            self.fc.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1]))
            if (i+1) < self.n_mlp_layers:
                self.bn.append(nn.BatchNorm1d(self.mlp_dims[i+1]))
                self.do.append(nn.Dropout(p=self.dropout_prop))

    def forward(self, x):
        """
        Input: (batch_size, n_points, in_dim).
        Outputs: (batch_size, n_classes).
        """
        x = x.transpose(2, 1)

        x = F.relu(self.n1(self.conv1(x)))
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))
        x = F.relu(self.n4(self.conv4(x)))
        x = F.relu(self.n5(self.conv5(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # x: (batch_size, out_dim)

        for i in range(self.n_mlp_layers):
            if (i+1) < self.n_mlp_layers:
                x = self.do[i](F.relu(self.bn[i](self.fc[i](x))))
            else:
                x = self.fc[i](x)

        # x: (batch_size, n_classes)
        return x
