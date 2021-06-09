import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from models.pixelcnn import PixelCNN


model = PixelCNN()
model.load_state_dict(torch.load(
    'checkpoints/Model_Checkpoint_'+'Last'+'.pt', map_location='cpu'))
model.eval()

sample = torch.Tensor(1, 1, 100, 106)
sample.fill_(0)

out = model(sample)
for i in range(100):
    for j in range(106):
        probs = F.softmax(out[:,:,i,j], dim=-1).data
        sample[:,:,i,j] = torch.multinomial(probs, 1).float()

torchvision.utils.save_image(sample, 'sample.png')