import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TeacherMLP, self).__init__()
        # 定义每一层
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        stem = self.relu(self.fc1(x))      # stem 对应第一层特征
        rb1  = self.relu(self.fc2(stem))   # rb1
        rb2  = self.relu(self.fc3(rb1))    # rb2
        rb3  = self.relu(self.fc4(rb2))    # rb3
        out  = self.fc5(rb3)               # logits
        feat = rb3                         # feat 用于 PKT/RKD/CC 等

        return stem, rb1, rb2, rb3, feat, out


class StudentMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(StudentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        stem = self.relu(self.fc1(x))
        rb1  = self.relu(self.fc2(stem))
        rb2  = self.relu(self.fc3(rb1))
        rb3  = self.relu(self.fc4(rb2))
        out  = self.fc5(rb3)
        feat = rb3

        return stem, rb1, rb2, rb3, feat, out