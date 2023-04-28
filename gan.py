from __future__ import print_function
from re import I

import torch
import torch.utils.data
from torch import conv2d, nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.empty(batch_size,noise_dim,device=device).normal_(mean=0,std=1)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
          nn.Linear(784, 400, bias=True, dtype=torch.float),
          nn.LeakyReLU(0.05),
          nn.Linear(400, 200, bias=True, dtype=torch.float),
          nn.LeakyReLU(0.05),
          nn.Linear(200, 100, bias=True, dtype=torch.float),
          nn.LeakyReLU(0.05),
          nn.Linear(100, 1, bias=True, dtype=torch.float)
        )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Linear(noise_dim, 128, bias=True, dtype=torch.float),
      nn.ReLU(),
      nn.Linear(128, 256, bias=True, dtype=torch.float),
      nn.ReLU(),
      nn.Linear(256, 512, bias=True, dtype=torch.float),
      nn.ReLU(),
      nn.Linear(512, 784, bias=True, dtype=torch.float),
      nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    loss_real = nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    loss_fake = nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    loss = loss_real + loss_fake
    return loss
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3,betas = (.05,.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10):
  """
  Train a GAN!
  
  Inputs:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - loader_train: the dataset used to train GAN
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  """
  iter_count = 0
  for epoch in range(num_epochs):
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      ##############################################################################
      # TODO: Implement an iteration of training the discriminator.                #
      # Replace 'pass' with your code.                                             #
      # Save the overall discriminator loss in the variable 'd_total_error',       #
      # which will be printed after every 'show_every' iterations.                 #
      #                                                                            #    
      # IMPORTANT: make sure to pre-process your real data (real images),          #
      # so as to make it in the range [-1,1].                                      #
      ##############################################################################
      d_total_error = 0
      noise = sample_noise(batch_size,noise_size,device=device)
      fake_images = G(noise)
      real_images = x.reshape(batch_size, -1).to(device)
      real_images = (real_images - 0.5) * 2
      logits_real = D(real_images)
      logits_fake = D(G(noise))
      d_total_error += discriminator_loss(logits_real, logits_fake)
      D_solver.zero_grad()
      d_total_error.backward(retain_graph=True)
      D_solver.step()
    
      ##############################################################################
      #                              END OF YOUR CODE                              #
      ##############################################################################        


      ##############################################################################
      # TODO: In the same iteration, implement training of the generator now   .   #
      # Replace 'pass' with your code.                                             #
      # Save the generator loss in the variable 'g_error', which will be printed.  #
      # after every 'show_every' iterations, and save the fake images generated    #
      # by G in the variable 'fake_images', which will be used to visualize the    #
      # generated images.
      ##############################################################################
      g_error = 0
      G_solver.zero_grad()
      g_error += generator_loss(logits_fake)
      g_error.backward(retain_graph=True)
      G_solver.step()
      ##############################################################################
      #                              END OF YOUR CODE                              #
      ############################################################################## 
  
      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      return imgs_numpy    




def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Unflatten(1, (1,28,28)),
      nn.Conv2d(1,32,(5,5),stride=1),
      nn.LeakyReLU(.01),
      nn.MaxPool2d((2,2),stride=2),
      nn.Conv2d(32,64,(5,5),stride=1),
      nn.LeakyReLU(.01),
      nn.MaxPool2d((2,2),stride=2),
      nn.Flatten(),
      nn.Linear(4*4*64,4*4*64),
      nn.LeakyReLU(.01),
      nn.Linear(4*4*64,1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Linear(NOISE_DIM,1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      nn.Linear(1024,7*7*128),
      nn.ReLU(),
      nn.BatchNorm1d(7*7*128),
      nn.Unflatten(1, (128,7,7)),
      nn.ConvTranspose2d(128,64,(4,4),stride=2,padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.ConvTranspose2d(64,1,(4,4),stride=2,padding=1),
      nn.Tanh(),
      nn.Flatten()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
