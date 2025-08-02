import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Simple UNet-like CNN for image-to-image tasks
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ChannelDataset(Dataset):
    # Loads (temp_map, channel_mask) pairs
    def __init__(self, root_dir):
        self.samples = []
        for subdir in os.listdir(root_dir):
            arr = np.load(os.path.join(root_dir, subdir, "temp_map.npy"))
            # Generate/Load optimal channel_mask.npy for each temp_map
            # Here, for demo, we use a dummy mask
            mask = (arr > arr.mean()).astype(np.float32)
            self.samples.append((arr[None, ...], mask[None, ...]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        temp_map, mask = self.samples[idx]
        return torch.from_numpy(temp_map).float(), torch.from_numpy(mask).float()

if __name__ == "__main__":
    dataset = ChannelDataset("data/")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleUNet()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        for temp_map, mask in dataloader:
            out = model(temp_map)
            loss = loss_fn(out, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "models/cool_chan_cnn.pth")
