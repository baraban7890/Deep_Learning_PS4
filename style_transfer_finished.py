"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from ps4_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

# 5 points
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "Pass" statement with your code
    #print("content_current.shape: ", content_current.shape)
    loss = nn.MSELoss(reduction='sum')
    if content_current.dim() == 4:
      _, C_l, H_l, W_l = content_current.shape
      M_l = H_l * W_l
      content_current = torch.squeeze(content_current, 0)
      content_original = torch.squeeze(content_original, 0)
      F = torch.reshape(content_current, (C_l, M_l))
      P = torch.reshape(content_original, (C_l, M_l))
      L = content_weight*loss(F,P)
    else: 
      N, R, C_l, H_l, W_l = content_current.shape
      M_l = H_l * W_l
      F = torch.reshape(content_current, (N, R, C_l, M_l))
      P = torch.reshape(content_original, (N, R, C_l, M_l))
      L = content_weight*loss(F, P)
    return L
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 9 points
def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "Pass" statement with your code

    N, C, H, W = features.shape
    norm = H*W*C
    M = H*W
    features = torch.reshape(features, (N,C,M))
    features_t = torch.transpose(features, 1,2)
    gram = torch.bmm(features, features_t)
    if normalize == True:
      gram = gram/norm

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram

# 9 points
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "Pass" statement with your code
    L = torch.zeros(1, dtype=feats[0].dtype, device=feats[0].device)
    loss = nn.MSELoss(reduction = 'sum')
    for i in range(len(style_layers)):
      layer_i = style_layers[i]
      feat_i = feats[layer_i]
      G_i = gram_matrix(feat_i)
      G_i = torch.squeeze(G_i, 0)
      L_i = loss(G_i, torch.squeeze(style_targets[i], 0))
      L_i = L_i*style_weights[i]
      L = L + L_i

    return L

    


    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 8 points
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "Pass" statement with your code
    loss = nn.MSELoss(reduction='sum')
    img = torch.squeeze(img, 0)
    img_h_top = img[:,1:,:]
    img_h_bot = img[:,:-1,:]
    img_w_top = img[:,:,1:]
    img_w_bot = img[:,:,:-1]
    loss_h = loss(img_h_top,img_h_bot)
    loss_w = loss(img_w_top, img_w_bot)
    L = tv_weight*(loss_h + loss_w)
    return L

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

# 10 points
def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "Pass" statement with your code

  #feat_0 = torch.empty_like(features, dtype=features.dtype, device=features.device)
  #print("features.shape: ", features.shape)
  #print("masks.shape: ", masks.shape)
  C = features.shape[2]
  for i in range(C):
    #print(i)
    #print('masks[:,0,:,:].shape: ', masks[:,0,:,:].shape)
    #print('features[:,0,i,:,:].shape: ', features[:,0,i,:,:].shape)
    #print('torch.mul(masks[:,0,:,:], features[:,0,i,:,:]).shape: ', torch.mul(masks[:,0,:,:], features[:,0,i,:,:]).shape)
    features[:,0,i,:,:] = torch.mul(masks[:,0,:,:], features[:,0,i,:,:])
    features[:,1,i,:,:] = torch.mul(masks[:,1,:,:], features[:,1,i,:,:])
  
  feat_0 = torch.squeeze(features[:,0,:,:], 1)
  feat_1 = torch.squeeze(features[:,1,:,:], 1)
  G_0 = gram_matrix(feat_0, normalize)
  G_1 = gram_matrix(feat_1, normalize)
  guided_gram = torch.stack((G_0, G_1), dim=1)
  return guided_gram
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

# 9 points
def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "Pass" statement with your code
    L = torch.zeros(1, dtype=feats[0].dtype, device=feats[0].device)
    loss = nn.MSELoss(reduction = 'sum')
    for i in range(len(style_layers)):
      layer_i = style_layers[i]
      feat_i = feats[layer_i]
      G_i = guided_gram_matrix(feat_i, content_masks[layer_i])
      G_i = torch.squeeze(G_i, 0)
      L_i = loss(G_i, torch.squeeze(style_targets[i], 0))
      L_i = L_i*style_weights[i]
      L = L + L_i

    return L
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
