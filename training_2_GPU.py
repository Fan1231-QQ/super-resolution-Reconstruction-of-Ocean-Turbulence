import os
os.environ['KERAS_BACKEND'] = 'jax'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error

import keras
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

import keras.ops as kops
import jax.numpy as jnp

from scaling import downscale_4D, upscale_4D

def pad_periodic_jax(x, n_pad_rows=0, n_pad_cols=0):
    r = n_pad_rows // 2
    c = n_pad_cols // 2
    top = x[:, -r:, :, :]
    bottom = x[:, :r, :, :]
    x = jnp.concatenate([top, x, bottom], axis=1)

    left = x[:, :, -c:, :]
    right = x[:, :, :c, :]
    x = jnp.concatenate([left, x, right], axis=2)
    return x

def periodic_convolution(x, n_filters, kernel_size, activation,
                         strides=(1, 1), n_pad_rows=0, n_pad_cols=0):
    def output_shape_fn(s):
        return (s[0], s[1] + n_pad_rows, s[2] + n_pad_cols, s[3])

    x = Lambda(lambda x: pad_periodic_jax(x, n_pad_rows, n_pad_cols),
               output_shape=output_shape_fn)(x)
    x = Conv2D(n_filters, kernel_size, activation=activation,
               padding='valid', strides=strides)(x)
    return x


# ========== Load and preprocess data ==========
file_path = "data_generation.h5"
beta, L_y = 2.0e-11, 1.8e6

with h5py.File(file_path, "r") as h5f:
    q_data = h5f["q"][:1000, ...]

q_data = q_data / (beta * L_y)

cgf = 16
downsample = downscale_4D(q_data, cgf)
upsample = upscale_4D(downsample, cgf)

upsampled_input = np.transpose(upsample, (0, 2, 3, 1))
raw = np.transpose(q_data, (0, 2, 3, 1))
residual = raw - upsampled_input

x = upsampled_input
y = residual

split_idx = int(0.8 * len(x))
x_train, x_val = x[:split_idx,...], x[split_idx:,...]
y_train, y_val = y[:split_idx,...], y[split_idx:,...]

# ========== Calculate Baseline Validation MSE ==========
upsample_val = x_val.reshape(len(x_val), -1)
raw_val = raw[split_idx:].reshape(len(x_val), -1)


# ========== Sequential Training Loop ==========
learning_rates = [1e-3, 5e-4, 1e-4]
histories = {}

n_layers = 4
for lr in learning_rates:
    print(f"\n--- Training with learning rate: {lr} ---")
    
    # reconstruct models
    input_tensor = keras.Input(shape=x.shape[1:])
    x_net = input_tensor
    for _ in range(n_layers):
        x_net = periodic_convolution(x_net, n_filters=32, kernel_size=5,
                                     activation='gelu', n_pad_rows=4, n_pad_cols=4)
    x_net = periodic_convolution(x_net, n_filters=2, kernel_size=5,
                                 activation='linear', n_pad_rows=4, n_pad_cols=4)
    model = keras.Model(inputs=input_tensor, outputs=x_net)

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=2000,
        batch_size=32,
        verbose=1
    )
    
    histories[lr] = history.history



# ========= Save Training Results to HDF5 ==========
with h5py.File("training_results.h5", "w") as h5f:
    for lr, hist in histories.items():
        grp = h5f.create_group(f"lr_{lr:.0e}")
        grp.create_dataset("loss", data=np.array(hist["loss"]))
        grp.create_dataset("val_loss", data=np.array(hist["val_loss"]))
        if "mae" in hist:
            grp.create_dataset("mae", data=np.array(hist["mae"]))
        if "val_mae" in hist:
            grp.create_dataset("val_mae", data=np.array(hist["val_mae"]))

print("Training results saved to training_results.h5")
