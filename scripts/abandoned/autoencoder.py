import numpy as np
import keras
from keras import layers, Model, Input

class Autoencoder:
    def __init__(self, epoch, hidden_dim=50):
        self.epoch = epoch
        self.epoch_length = epoch.shape[0]
        self.hidden_dim = hidden_dim
        self.autoencoder, self.encoder, self.decoder = self.build_autoencoder()

    def build_autoencoder(self):
        # Input layer
        input = Input(shape=(self.epoch_length,))
        
        # Encoded input
        encoded = layers.Dense(self.hidden_dim, activation=keras.ops.log_sigmoid)(input)
        
        # Decoded reconstruction
        decoded = layers.Dense(self.epoch_length)(encoded)
        
        # Autoencoder
        autoencoder = Model(input, decoded)

        # Encoder
        encoder = Model(input, encoded)
        encoded_input = Input(shape=(self.hidden_dim,))

        # Decoder
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        return autoencoder, encoder, decoder

    def train(self, epochs=500, batch_size=32):
        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # Configure EarlyStopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5)
        
        # Train the model
        self.autoencoder.fit(self.epoch,
                             self.epoch,
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[early_stopping],
                             )
        
        return self.autoencoder
    
    def get_weights(self):
        return self.autoencoder.get_weights()