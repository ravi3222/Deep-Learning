# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as penalties
import os
import pickle
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape
from keras.layers import Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

class VariationalAutoencoder():

  def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, 
               encoder_conv_strides, decoder_conv_t_filters, decoder_conv_t_kernel_size, 
               decoder_conv_t_strides, z_dim, use_dropout= False):

    self.name = 'variational_autoencoder'
    
    self.input_dim = input_dim # size of input image
    self.encoder_conv_filters = encoder_conv_filters # encoder conv layers depth
    self.encoder_conv_kernel_size = encoder_conv_kernel_size # encoder conv kernel size
    self.encoder_conv_strides = encoder_conv_strides # encoder conv strides
    self.decoder_conv_t_filters = decoder_conv_t_filters # decoder conv transpose layers depth
    self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size # decoder conv kernel size
    self.decoder_conv_t_strides = decoder_conv_t_strides # decoder conv strides
    self.z_dim = z_dim # dimension of latent space
    self.use_dropout = use_dropout # use dropouts or not
    
    self.n_layers_encoder = len(encoder_conv_filters) # number of encoder conv layers
    self.n_layers_decoder = len(decoder_conv_t_filters) # number of decoder conv transpose layers
    
    pdb.set_trace()
    self._build()

  
  ## BUILD THE FULL VAE MODEL
  def _build(self):
    
    # THE ENCODER 
    # A model that takes an input image and encodes it into the 2D latent space, 
    # by sampling a point from the normal distribution defined by mu and log_var.”

    encoder_input = Input(shape=self.input_dim, name='encoder_input')
    x = encoder_input
    
    for i in range(self.n_layers_encoder):
      conv_layer = Conv2D(filters = self.encoder_conv_filters[i], 
                          kernel_size = self.encoder_conv_kernel_size[i], 
                          strides = self.encoder_conv_strides[i], 
                          padding = 'same', name = 'encoder_conv_' + str(i))
      x = conv_layer(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      if self.use_dropout:
        x = Dropout(rate = 0.25)(x)
      
    shape_before_flattening = K.int_shape(x)[1:]
    x = Flatten()(x)
    self.mu = Dense(self.z_dim, name='mu')(x)
    self.log_var = Dense(self.z_dim, name='log_var')(x)
    # We choose to map to the logarithm of the variance, as this can take any real 
    # number in the range (–inf, inf), matching the natural output range from a 
    # neural network unit, whereas variance values are always positive.

    self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

    # Now, since we are sampling a random point from an area around mu, the decoder 
    # must ensure that all points in the same neighborhood produce very similar images when 
    # decoded, so that the reconstruction loss remains small.

    def sampling(args):
      mu, log_var = args
      epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
      return mu + K.exp(log_var / 2) * epsilon

    # Latent space
    encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])
    
    self.encoder = Model(encoder_input, encoder_output)

    # THE DECODER
    # A model that takes a point in the latent space and decodes it into the original image domain.

    decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    for i in range(self.n_layers_decoder):
      conv_t_layer = Conv2DTranspose(filters = self.decoder_conv_t_filters[i], 
                                     kernel_size = self.decoder_conv_t_kernel_size[i], 
                                     strides = self.decoder_conv_t_strides[i], 
                                     padding = 'same', name = 'decoder_conv_t_' + str(i))

      x = conv_t_layer(x)
      if i < self.n_layers_decoder - 1: # condition for not having bn-leakyrelu-dropout at last layer
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
          x = Dropout(rate = 0.25)(x)
      else:
        x = Activation('sigmoid')(x)

    decoder_output = x
    self.decoder = Model(decoder_input, decoder_output)

    ### THE FULL VAE
    model_input = encoder_input
    model_output = self.decoder(encoder_output)

    self.model = Model(model_input, model_output)

  ## DEFINE THE LOSS FUNCTIONS AND OPTIMIZER
  def compile(self, learning_rate, reco_loss_factor):
    self.learning_rate = learning_rate
    # Binary cross-entropy places heavier penalties on predictions at the extremes 
    # that are badly wrong, so it tends to push pixel predictions to the middle of the 
    # range. This results in less vibrant images. For this reason, we use RMSE as the 
    # loss function.

    def vae_r_loss(y_true, y_pred):
      r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
      return reco_loss_factor * r_loss # reco_loss_factor ensures balance with the KL divergence loss

    # KL divergence term penalizes the network for encoding observations to mu 
    # and log_var variables that differ significantly from the parameters of a 
    # standard normal distribution, namely mu = 0 and log_var = 0.

    def vae_kl_loss(y_true, y_pred):
      kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
      return kl_loss

    def vae_loss(y_true, y_pred):
      reco_loss = vae_r_loss(y_true, y_pred)
      kl_loss = vae_kl_loss(y_true, y_pred)
      return  reco_loss + kl_loss

    optimizer = Adam(lr=learning_rate)
    self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
    
INPUT_DIM = (128,128,3)

vae = VariationalAutoencoder(input_dim = INPUT_DIM,
							 encoder_conv_filters = [32,64,64,64],
							 encoder_conv_kernel_size = [3,3,3,3],
							 encoder_conv_strides = [2,2,2,2],
							 decoder_conv_t_filters = [64,64,32,3],
							 decoder_conv_t_kernel_size = [3,3,3,3],
							 decoder_conv_t_strides = [2,2,2,2],
							 z_dim=200,
							 use_dropout=True)

vae.encoder.summary()
vae.decoder.summary()

