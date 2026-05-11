from torch.nn.modules.dropout import Dropout
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def configure_optimizer(model: nn.Module,lr=0.001,weight_decay=0.000222) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
class EnhancedCNN(nn.Module):
    def __init__(self, conv_dropout=0.118, dropout=0.573):
        super(EnhancedCNN, self).__init__()

        # Bloc 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(64)
        self.pool1   = nn.MaxPool2d(2)
        self.drop1   = nn.Dropout(conv_dropout)

        # Bloc 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2   = nn.BatchNorm2d(128)
        self.pool2   = nn.MaxPool2d(2)
        self.drop2   = nn.Dropout(conv_dropout * 1.5)

        # Bloc 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(256)
        self.pool3   = nn.MaxPool2d(2)
        self.drop3   = nn.Dropout(conv_dropout * 2.0)

        # Bloc 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1   = nn.BatchNorm2d(512)
        self.pool4   = nn.MaxPool2d(2)

        # Pooling global
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(512, 256)
        self.drop_fc  = nn.Dropout(dropout)
        self.fc2      = nn.Linear(256, 2)

    def forward(self, x):
        print(f"INPUT:          {x.shape}")  # (batch, 1, 128, 157)

        # Bloc 1
        x = self.conv1_1(x);  print(f"Conv1_1:        {x.shape}")  # (batch, 64, 128, 157)
        x = self.bn1_1(x);    print(f"BN1_1:          {x.shape}")  # (batch, 64, 128, 157)
        x = F.relu(x);        print(f"ReLU:           {x.shape}")  # (batch, 64, 128, 157)
        x = self.conv1_2(x);  print(f"Conv1_2:        {x.shape}")  # (batch, 64, 128, 157)
        x = self.bn1_2(x);    print(f"BN1_2:          {x.shape}")  # (batch, 64, 128, 157)
        x = F.relu(x);        print(f"ReLU:           {x.shape}")  # (batch, 64, 128, 157)
        x = self.pool1(x);    print(f"MaxPool1:       {x.shape}")  # (batch, 64,  64,  78)
        x = self.drop1(x);    print(f"Dropout1:       {x.shape}")  # (batch, 64,  64,  78)

        # Bloc 2
        x = self.conv2_1(x);  print(f"Conv2_1:        {x.shape}")  # (batch, 128, 64, 78)
        x = self.bn2_1(x);    print(f"BN2_1:          {x.shape}")
        x = F.relu(x);        print(f"ReLU:           {x.shape}")
        x = self.conv2_2(x);  print(f"Conv2_2:        {x.shape}")  # (batch, 128, 64, 78)
        x = self.bn2_2(x);    print(f"BN2_2:          {x.shape}")
        x = F.relu(x);        print(f"ReLU:           {x.shape}")
        x = self.pool2(x);    print(f"MaxPool2:       {x.shape}")  # (batch, 128, 32, 39)
        x = self.drop2(x);    print(f"Dropout2:       {x.shape}")

        # Bloc 3
        x = self.conv3_1(x);  print(f"Conv3_1:        {x.shape}")  # (batch, 256, 32, 39)
        x = self.bn3_1(x);    print(f"BN3_1:          {x.shape}")
        x = F.relu(x);        print(f"ReLU:           {x.shape}")
        x = self.conv3_2(x);  print(f"Conv3_2:        {x.shape}")  # (batch, 256, 32, 39)
        x = self.bn3_2(x);    print(f"BN3_2:          {x.shape}")
        x = F.relu(x);        print(f"ReLU:           {x.shape}")
        x = self.pool3(x);    print(f"MaxPool3:       {x.shape}")  # (batch, 256, 16, 19)
        x = self.drop3(x);    print(f"Dropout3:       {x.shape}")

        # Bloc 4
        x = self.conv4_1(x);  print(f"Conv4_1:        {x.shape}")  # (batch, 512, 16, 19)
        x = self.bn4_1(x);    print(f"BN4_1:          {x.shape}")
        x = F.relu(x);        print(f"ReLU:           {x.shape}")
        x = self.pool4(x);    print(f"MaxPool4:       {x.shape}")  # (batch, 512,  8,  9)

        # Pooling global
        x = self.adaptive_pool(x); print(f"AdaptivePool:  {x.shape}")  # (batch, 512, 1, 1)

        # Classifier
        x = self.flatten(x);  print(f"Flatten:        {x.shape}")  # (batch, 512)
        x = self.fc1(x);      print(f"FC1:            {x.shape}")  # (batch, 256)
        x = F.relu(x);        print(f"ReLU:           {x[0][0:5]}")  # (batch, 256)
        x = self.drop_fc(x);  print(f"Dropout FC:     {x}")  # (batch, 256)
        x = self.fc2(x);      print(f"FC2 (OUTPUT):   {x.shape}")  # (batch, 2)
        print(x)

        return x

model = EnhancedCNN()
model.eval()  #deactivate dropout

x = np.random.rand(80000)  # 1 audio de 5 secunde
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axes = axes.flatten()

mel_spec = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=2048,hop_length=512, n_mels=128) #shape=(128,157)
print(mel_spec.shape)
mfcc_spec=librosa.feature.mfcc(S=librosa.power_to_db(mel_spec, ref=np.max), n_mfcc=20, sr=16000, n_fft=2048,hop_length=512, n_mels=128) #shape=(128,157)
print(mfcc_spec.shape)
axes[0].imshow( #mfcc spectrogram
    mfcc_spec,
    origin='lower',
    aspect='auto',
    cmap='magma'
)
axes[0].set_title("MFCC Spectrogram")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("MFCC bins")

axes[1].imshow( #mel spectrogram
    mel_spec,
    origin='lower',
    aspect='auto',
    cmap='magma'
)
axes[1].set_title("Mel Spectrogram")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Mel bins")

axes[2].imshow( #log-mel spectrogram
    librosa.power_to_db(mel_spec, ref=np.max),
    origin='lower',
    aspect='auto',
    cmap='magma'
)
axes[2].set_title("Log MelSpectrogram")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Mel bins")

# plt.tight_layout()
# plt.show()
x=torch.rand(1,1,128,157)
with torch.no_grad():
    model(x)
