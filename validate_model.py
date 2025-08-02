import torch
import numpy as np
from train_cnn import SimpleUNet

def model_predict(temp_map_path, model_path="models/cool_chan_cnn.pth"):
    temp_map = np.load(temp_map_path)
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    inp = torch.from_numpy(temp_map[None, None, ...]).float()
    with torch.no_grad():
        mask = model(inp).squeeze().numpy()
    np.save(temp_map_path.replace("temp_map.npy", "pred_channel.npy"), mask)
    return mask

if __name__ == "__main__":
    mask = model_predict("data/t_load_1.00/temp_map.npy")
    print("Predicted channel mask shape:", mask.shape)
