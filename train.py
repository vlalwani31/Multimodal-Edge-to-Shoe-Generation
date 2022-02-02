import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb

# Training Configurations
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = './edges2shoes/train/'
test_img_dir = './edges2shoes/val/'
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs = 20
batch_size = 16
lr_rate = 0.0002  	  # Adam optimizer learning rate
betas = (0.5, 0.999)  # Adam optimizer beta 1, beta 2
lambda_pixel = 15       # Loss weights for L1 image loss
lambda_latent = 50      # Loss weights for L1 latent regression
lambda_kl = 0.01          # Loss weights for kl divergence
latent_dim = 8         # latent dimension for the encoded images from domain B
gpu_id = 0

# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

# Reparameterization helper function
# (You may need this helper function here or inside models.py, depending on your encoder implementation)
def reparameterization(mean, log_var):
    ################################
    # Please fill in std, eps and z:
    std = torch.exp(0.5 * log_var)
    eps = torch.normal(0,1,size=std.shape,device=std.device, requires_grad=False)
    z = mean + (std * eps)
    ################################

    return z

def init_weights(m):
  if isinstance(m, nn.Conv2d):
    nn.init.xavier_uniform(m.weight)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.normal_(m.weight.data, 0.0, 0.02)
    nn.init.constant_(m.bias.data, 0.01)

def set_requires_grad(nets, requires_grad=False):
  for net in nets:
    if net is not None:
      for param in net.parameters():
        param.requires_grad = requires_grad




# Random seeds (optional)
torch.manual_seed(1); np.random.seed(1)
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize networks
generator = Generator(latent_dim, img_shape).to(device)
encoder = Encoder(latent_dim).to(device)
D_VAE = Discriminator().to(device)
D_CLR = Discriminator().to(device)

# Initialize Weights for networks
generator.apply(init_weights)
D_VAE.apply(init_weights)
D_CLR.apply(init_weights)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate/40, betas=betas)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate*4, betas=betas)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate/500, betas=betas)
optimizer_D_CLR = torch.optim.Adam(D_CLR.parameters(), lr=lr_rate/500, betas=betas)

# Define DataLoader
dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
test_img_dir = './edges2shoes/val/'
test_dataset = Edge2Shoe(test_img_dir)
test_loader = data.DataLoader(test_dataset, batch_size=1)

# Losses for networks
L1_VAE_Loss = nn.L1Loss(reduction='sum')
L1_CRL_Loss = nn.L1Loss(reduction='sum')
Dis_loss = nn.BCELoss(reduction='sum')
def KLD_VAE_Loss(mu, log_var):
  std = torch.exp(log_var)
  KLD = torch.sum(std + torch.square(mu) - log_var - 1) / 2
  return KLD

# For adversarial loss (optional to use)
valid = 1; fake = 0
# Fixed input to test the performance of GAN
fixed_noise = torch.randn(36, latent_dim, device=device)
for test_idx, test_data in enumerate(test_loader):
  test_edge_tensor, test_rgb_tensor = test_data
  test_edge_tensor = test_edge_tensor.repeat(36,1,1,1)
  test_edge_tensor = norm(test_edge_tensor).to(device)
  break

GE_losses = []
D_VAE_losses = []
D_CLR_losses = []
L1_VAE_losses = []
G_VAE_losses = []
G_CLR_losses = []
KL_VAE_losses = []
L1_CLR_losses = []
img_list = []
len_dataset = len(loader)

for e in range(num_epochs):
    for idx, datat in enumerate(loader):
      edge_tensor, rgb_tensor = datat
      len_batch = edge_tensor.shape[0]
      len_data = len_batch // 2
      # Send the data to device
      edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
      # Split the data for vae and clr
      if (len_data == 0):
        real_A_vae = edge_tensor
        real_A_clr = edge_tensor
        real_B_vae = rgb_tensor
        real_B_clr = rgb_tensor
        Valid_label_vae = torch.full((1,1,1,1), valid, dtype=torch.float, device=device, requires_grad=False)
        Fake_label_vae = torch.full((1,1,1,1), fake, dtype=torch.float, device=device, requires_grad=False)
        Valid_label_clr = torch.full((1,1,1,1), valid, dtype=torch.float, device=device, requires_grad=False)
        Fake_label_clr = torch.full((1,1,1,1), fake, dtype=torch.float, device=device, requires_grad=False)
      else:
        real_A_vae = edge_tensor[0:len_data,:,:,:]
        real_A_clr = edge_tensor[len_data:,:,:,:]
        real_B_vae = rgb_tensor[0:len_data,:,:,:]
        real_B_clr = rgb_tensor[len_data:,:,:,:]
        Valid_label_vae = torch.full((len_data,1,1,1), valid, dtype=torch.float, device=device, requires_grad=False)
        Fake_label_vae = torch.full((len_data,1,1,1), fake, dtype=torch.float, device=device, requires_grad=False)
        Valid_label_clr = torch.full(((len_batch - len_data),1,1,1), valid, dtype=torch.float, device=device, requires_grad=False)
        Fake_label_clr = torch.full(((len_batch - len_data),1,1,1), fake, dtype=torch.float, device=device, requires_grad=False)

      #-------------------------------
      #  Train Generator and Encoder
      #------------------------------

      # Pass B through Encoder for VAE
      encoder_output_mean, encoder_output_logvar = encoder(real_B_vae)
      z = reparameterization(encoder_output_mean, encoder_output_logvar)
      # z = z.detach()
      # Get a random normal tensor for CLR
      z_clr = torch.randn(len_batch - len_data, latent_dim, device=device, requires_grad=False)
      # Generate fake B for VAE with A and z
      fake_B_vae = generator(real_A_vae, z)
      # Generate fake B for CLR with A and z_clr
      fake_B_clr = generator(real_A_clr, z_clr)
      # Pass fake B through Encoder
      fake_mu, fake_logvar = encoder(fake_B_clr)


      # Assign all the descriminator to not calculate gradients
      set_requires_grad([D_VAE, D_CLR], False)
      # Clear Optimizers
      optimizer_E.zero_grad()
      optimizer_G.zero_grad()
      # Calculate all the losses
      loss_L1_vae = L1_VAE_Loss(fake_B_vae, real_B_vae) * lambda_pixel
      loss_G_vae = Dis_loss(D_VAE(fake_B_vae), Valid_label_vae) * 2
      loss_G_clr = Dis_loss(D_CLR(fake_B_clr), Valid_label_clr) * 2
      loss_kl_vae = KLD_VAE_Loss(encoder_output_mean, encoder_output_logvar) * lambda_kl

      # Calculate the total loss
      total_loss = loss_G_vae + loss_G_clr + loss_kl_vae + loss_L1_vae
      # Backtrack it
      total_loss.backward(retain_graph=True)

      # Assign the encoder to not calculate gradients
      set_requires_grad([encoder], False)
      # Calculate loss for latent dim in CLR
      loss_L1_clr = L1_CRL_Loss(fake_mu, z_clr) * lambda_latent
      loss_L1_clr.backward()
      # Assign the encoder to again calculate gradients
      set_requires_grad([encoder], True)

      # Take a step for Optimizers
      optimizer_E.step()
      optimizer_G.step()

      # if (e % 3 == 0):
      #----------------------------------
      #  Train Discriminator (cVAE-GAN)
      #----------------------------------

      # Assign all the descriminator to not calculate gradients
      set_requires_grad([D_VAE, D_CLR], True)
      optimizer_D_VAE.zero_grad()
      # Calculate output for generated B (with detach from generator) and Real B
      pred_fake = D_VAE(fake_B_vae.detach())
      pred_real = D_VAE(real_B_vae)
      # Calculate Loss
      loss_D_vae_fake = Dis_loss(pred_fake, Fake_label_vae + 0.2)
      loss_D_vae_real = Dis_loss(pred_real, Valid_label_vae - 0.2)
      # Sum the losses and do backward
      loss_D_vae = (loss_D_vae_fake + loss_D_vae_real) / 2
      loss_D_vae.backward()
      # Take a step for Optimizer
      optimizer_D_VAE.step()

      #---------------------------------
      #  Train Discriminator (cLR-GAN)
      #---------------------------------

      optimizer_D_CLR.zero_grad()
      # Calculate output for generated B (with detach from generator) and Real B
      pred_fake_clr = D_CLR(fake_B_clr.detach())
      pred_real_clr = D_CLR(real_B_clr)
      # Calculate Loss
      loss_D_clr_fake = Dis_loss(pred_fake_clr, Fake_label_clr + 0.2)
      loss_D_clr_real = Dis_loss(pred_real_clr, Valid_label_clr - 0.2)
      # Sum the losses and do backward
      loss_D_clr = (loss_D_clr_fake + loss_D_clr_real) / 2
      loss_D_clr.backward()
      # Take a step for Optimizer
      optimizer_D_CLR.step()
      # else:
      #   loss_D_vae = torch.tensor(0)
      #   loss_D_clr = torch.tensor(0)


      # Save information
      if idx % 150 == 0:
        print("[",e,"/",num_epochs,"][",idx,"/",len_dataset,"]\tLoss_D_clr: ",loss_D_clr.item(),"\t",loss_D_vae.item(), "\tLoss_GE: ",loss_L1_clr.item(), "\t", loss_L1_vae.item(), "\t", loss_kl_vae.item(), "\t", loss_G_vae.item(), "\t", loss_G_clr.item())
        GE_losses.append(total_loss.item())
        D_VAE_losses.append(loss_D_vae.item())
        D_CLR_losses.append(loss_D_clr.item())
        L1_VAE_losses.append(loss_L1_vae.item())
        G_VAE_losses.append(loss_G_vae.item())
        G_CLR_losses.append(loss_G_clr.item())
        KL_VAE_losses.append(loss_kl_vae.item())
        L1_CLR_losses.append(loss_L1_clr.item())



      if ((e % 1 == 0) or (e == num_epochs-1)) and (idx == 0):
        with torch.no_grad():
          gen_output = generator(test_edge_tensor, fixed_noise)
        grid_gen_output = vutils.make_grid(denorm(gen_output), nrow=6, padding=2, normalize=True)
        img_list.append(grid_gen_output)
        # plt.imshow(np.transpose(grid_gen_output.cpu().detach().numpy(), (1, 2, 0)))
        # plt.title("Gan's output at epoch: " + str(e))
        # plt.show()

checkpoint = {'netG': generator.state_dict(),
              'netE': encoder.state_dict(),
              'netD1': D_VAE.state_dict(),
              'netD2': D_CLR.state_dict()}
torch.save(checkpoint, './bicyclegan8_{}.pt'.format(num_epochs))

# Display Discriminator losses
plt.title("Discriminator_VAE and Discriminator_CLR Loss During Training")
plt.plot(D_VAE_losses,label="D_vae")
plt.plot(D_CLR_losses,label="D_clr")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_D_loss.png")
# Display Total Generator-Encoder Losses
plt.title("Generator-Encoder Loss During Training")
plt.plot(GE_losses,label="GE")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_total_loss_loss.png")
# Display L1 VAE Loss
plt.title("L1 VAE Loss During Training")
plt.plot(L1_VAE_losses,label="L1 VAE")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_L1_vae_loss.png")
# Display L1 CLR Loss
plt.title("L1 CLR Loss During Training")
plt.plot(L1_CLR_losses,label="L1 CLR")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_L1_clr_loss.png")
# Display KL VAE Loss
plt.title("KL VAE Loss During Training")
plt.plot(KL_VAE_losses,label="KL")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_kl_vae_loss.png")
# Display G VAE Loss
plt.title("G VAE Loss During Training")
plt.plot(G_VAE_losses,label="G VAE")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_g_vae_loss.png")
# Display G CLR Loss
plt.title("G CLR Loss During Training")
plt.plot(G_CLR_losses,label="G CLR")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("gan_g_clr_loss.png")


for i,image_item in enumerate(img_list):
  plt.imshow(np.transpose(image_item.cpu().detach().numpy(), (1, 2, 0)))
  plt.title("Gan's output at epoch: " + str(i+1) + ".png")
  plt.savefig("gan_output_at_epoch_" + str(i+1) + ".png")
