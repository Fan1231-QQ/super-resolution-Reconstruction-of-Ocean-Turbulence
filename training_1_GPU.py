import os
os.environ['KERAS_BACKEND'] = 'jax'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import keras
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, Input, Lambda, UpSampling2D,Cropping2D
from keras.models import Model
from keras.optimizers import Adam
from keras import mixed_precision

import keras.ops as kops
import jax.numpy as jnp

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
    # 使用 Lambda 包装 tf 版本 padding
    def output_shape_fn(s):
        return (s[0], s[1] + n_pad_rows, s[2] + n_pad_cols, s[3])

    x = Lambda(lambda x: pad_periodic_jax(x, n_pad_rows, n_pad_cols),
               output_shape=output_shape_fn)(x)
    x = Conv2D(n_filters, kernel_size, activation=activation,
               padding='valid', strides=strides)(x)
    return x


# def residual_block_periodic_conv(x, kernel=(5,5), strides=(1,1),
#                                   n_pad_rows=4, n_pad_cols=4, activation='gelu'):
#     layer_input = x
#     n_filters = x.shape[-1]

#     x = BatchNormalization()(x)
#     x = Activation(activation)(x)
#     x = periodic_convolution(x, n_filters, kernel, strides=strides,
#                               n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation=activation)

#     x = BatchNormalization()(x)
#     x = periodic_convolution(x, n_filters, kernel, strides=strides,
#                               n_pad_rows=n_pad_rows, n_pad_cols=n_pad_cols, activation='linear')

#     x = keras.layers.add([x, layer_input])
#     return x

# ========== Load and preprocess data ==========
file_path = "data_generation.h5"
beta, L_y = 2.0e-11, 1.8e6

with h5py.File(file_path, "r") as h5f:
    q_data = h5f["q"][:1000, ...]

q_data = q_data / (beta * L_y)


from scaling import downscale_4D,upscale_4D
'''downsample and upsample'''

cgf=16
downsample=downscale_4D(q_data,cgf)
upsample=upscale_4D(downsample,cgf)


upsampled_input = np.transpose(upsample, (0, 2, 3, 1))
raw = np.transpose(q_data, (0, 2, 3, 1))


# 构建残差目标
residual = raw - upsampled_input


# ========== Build and train model ==========

# model = build_periodic_cnn(input_shape=(512, 512, 2), n_layers=4)

x = upsampled_input
y = residual

# ====== Step 2: Train/Val Split ======
split_idx = int(0.8 * len(x))
x_train, x_val = x[:split_idx,...], x[split_idx:,...]
y_train, y_val = y[:split_idx,...], y[split_idx:,...]



# ====== Step 3: Basic CNN (Flax style with Keras-JAX) ======

# Version2: simple CNN+ padding
input_tensor = Input(shape=x.shape[1:])  # (H, W, 2)
x_net = input_tensor  

n_layers = 4
for i in range(n_layers):
    act = 'gelu' 
    x_net = periodic_convolution(x_net, n_filters=64, kernel_size=5,
                              activation=act, n_pad_rows=4, n_pad_cols=4)

# Output layer
x_net = periodic_convolution(x_net, n_filters=2, kernel_size=5,
                         activation='linear', n_pad_rows=4, n_pad_cols=4)

model = keras.Model(inputs=input_tensor, outputs=x_net)
model.summary()


# _ = model(x_train[:1])  # 关键：手动构建模型
# ====== Step 4: Compile & Train ======
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# model.fit(x_train,y_train)
model.fit(x_train,y_train,
          validation_data=(x_val, y_val),
          epochs=100,
          batch_size=8,
          verbose=2)


y_pred = model.predict(x_val)  # 或者 x_test
#save data
import h5py
import numpy as np

# 保存路径
save_path = "prediction_results.h5"

# 创建并写入 HDF5 文件
with h5py.File(save_path, "w") as h5f:
    h5f.create_dataset("y_pred", data=y_pred)     # 预测残差
    h5f.create_dataset("x_val", data=x_val)       # 上采样输入
    h5f.create_dataset("raw", data=raw[split_idx:])  # 对应的原始真值（注意取验证集对应部分）

print(f"结果已保存为 HDF5 文件：{save_path}")

