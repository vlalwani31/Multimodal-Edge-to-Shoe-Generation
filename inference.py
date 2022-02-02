from torch.utils import data
from PIL import Image
import numpy as np
import torch
import glob
import pdb
import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.utils as vutils
import cv2
import plotly
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
from datasets import *
from models import *


# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_shape = (3, 128, 128)
img_dir = './edges2shoes/val/'
test_dataset = Edge2Shoe(test_img_dir)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)
model_saved = torch.load('./bicyclegan8_20.pt')
test_generator = Generator(latent_dim, img_shape).to(device)
test_generator.load_state_dict(model_saved['netG'])
test_generator.eval()
fixed_noise = torch.randn(36, latent_dim, device=device)
for test_idx, test_data in enumerate(test_loader):
  test_edge_tensor, test_rgb_tensor = test_data
  test_edge_tensor = test_edge_tensor.repeat(36,1,1,1)
  test_edge_tensor = norm(test_edge_tensor).to(device)
  break

plt.imshow(np.transpose(denorm(test_edge_tensor[0,:,:,:]).cpu().detach().numpy().astype(np.uint8), (1, 2, 0)))
plt.title("original edge image")
plt.show()
plt.imshow(test_rgb_tensor[0,:,:,:].cpu().numpy().astype(np.uint8).transpose((1,2,0)))
plt.title("original rgb image")
plt.show()
with torch.no_grad():
    gen_output = test_generator(test_edge_tensor, fixed_noise)
    sq_out = vutils.make_grid(denorm(gen_output), nrow=6, padding=2, normalize=True)
    plt.figure(figsize=(12,12))
    plt.imshow(np.transpose(sq_out.cpu().detach().numpy(), (1, 2, 0)))
    plt.show()
