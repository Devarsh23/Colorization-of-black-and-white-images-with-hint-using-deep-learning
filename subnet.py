#SubNet.py

__INITIALIZER__ = tf.random_normal_initializer(0., 0.02)
__MOMENTUM__ = 0.9
__EPSILON__ = 1e-5


def res_net_block_v2(inputs, filters):
    with tf.name_scope("ResNetBlock"):
        shortcut = inputs
        tensor = tf.keras.layers.BatchNormalization()(inputs)
        tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)

        tensor = tf.keras.layers.BatchNormalization()(tensor)
        tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)
        tensor = tf.keras.layers.add([shortcut, tensor])
    return tensor


def GenConvBlock(inputs, filters, k, s, res_net_block=True, name="GenConvBlock"):
    filters = int(filters)
    with tf.name_scope(name):
        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                        padding="SAME", kernel_initializer=__INITIALIZER__)(inputs)

        if res_net_block:
            tensor = res_net_block_v2(tensor, filters)
        else:
            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)
            tensor = tf.keras.layers.LeakyReLU()(tensor)

        return tensor


def GenUpConvBlock(inputs_a, inputs_b, filters, k, s, res_net_block=True, name="GenUpConvBlock"):
    filters = int(filters)
    with tf.name_scope(name):
        tensor = tf.keras.layers.Concatenate(3)([inputs_a, inputs_b])
        tensor = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=k, strides=s, use_bias=False,
                                                 padding="SAME", kernel_initializer=__INITIALIZER__)(tensor)

        if res_net_block:
            tensor = res_net_block_v2(tensor, filters)
        else:
            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)
            tensor = tf.keras.layers.ReLU()(tensor)

        return tensor


class DisConvBlock(tf.keras.Model):
    def __init__(self, filters, k, s, apply_bat_norm=True, name=None):
        super(DisConvBlock, self).__init__(name=name)
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.apply_bat_norm = apply_bat_norm
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s,
                                           padding="SAME", kernel_initializer=initializer)
        if self.apply_bat_norm:
            self.bn = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)

        self.act = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training):
        tensor = self.conv(inputs)

        if self.apply_bat_norm:
            tensor = self.bn(tensor, training=training)

        tensor = self.act(tensor)
        return tensor


def tf_int_round(num):
    return tf.cast(tf.round(num), dtype=tf.int32)


class resize_layer(tf.keras.layers.Layer):
    def __init__(self, size=(512, 512), **kwargs, ):
        super(resize_layer, self).__init__(**kwargs)
        (self.height, self.width) = size

    def build(self, input_shape):
        super(resize_layer, self).build(input_shape)

    def call(self, x, method="nearest"):
        height = 512
        width = 512

        if method == "nearest":
            return tf.image.resize_nearest_neighbor(x, size=(height, width))
        elif method == "bicubic":
            return tf.image.resize_bicubic(x, size=(height, width))
        elif method == "bilinear":
            return tf.image.resize_bilinear(x, size=(height, width))

    def get_output_shape_for(self, input_shape):
        return (self.input_shape[0], 512, 512, 3)


#PaintsTensorflow
def Generator(inputs_size=None, res_net_block=True, name="PaintsTensorFlow"):
    inputs_line = tf.keras.Input(shape=[inputs_size, inputs_size, 1], dtype=tf.float32, name="inputs_line")
    inputs_hint = tf.keras.Input(shape=[inputs_size, inputs_size, 3], dtype=tf.float32, name="inputs_hint")
    tensor = tf.keras.layers.Concatenate(3)([inputs_line, inputs_hint])

    e0 = GenConvBlock(tensor,gf_dim / 2, 3, 1, res_net_block=res_net_block, name="E0")  # 64
    e1 = GenConvBlock(e0, gf_dim * 1, 4, 2, res_net_block=res_net_block, name="E1")
    e2 = GenConvBlock(e1, gf_dim * 1, 3, 1, res_net_block=res_net_block, name="E2")
    e3 = GenConvBlock(e2, gf_dim * 2, 4, 2, res_net_block=res_net_block, name="E3")
    e4 = GenConvBlock(e3, gf_dim * 2, 3, 1, res_net_block=res_net_block, name="E4")
    e5 = GenConvBlock(e4, gf_dim * 4, 4, 2, res_net_block=res_net_block, name="E5")
    e6 = GenConvBlock(e5, gf_dim * 4, 3, 1, res_net_block=res_net_block, name="E6")
    e7 = GenConvBlock(e6, gf_dim * 8, 4, 2, res_net_block=res_net_block, name="E7")
    e8 = GenConvBlock(e7, gf_dim * 8, 3, 1, res_net_block=res_net_block, name="E8")

    d8 = GenUpConvBlock(e7, e8, gf_dim * 8, 4, 2, res_net_block=res_net_block, name="D8")
    d7 = GenConvBlock(d8, gf_dim * 4, 3, 1, res_net_block=res_net_block, name="D7")
    d6 = GenUpConvBlock(e6, d7, gf_dim * 4, 4, 2, res_net_block=res_net_block, name="D6")
    d5 = GenConvBlock(d6, gf_dim * 2, 3, 1, res_net_block=res_net_block, name="D5")
    d4 = GenUpConvBlock(e4, d5, gf_dim * 2, 4, 2, res_net_block=res_net_block, name="D4")
    d3 = GenConvBlock(d4, gf_dim * 1, 3, 1, res_net_block=res_net_block, name="D3")
    d2 = GenUpConvBlock(e2, d3, gf_dim * 1, 4, 2, res_net_block=res_net_block, name="D2")
    d1 = GenConvBlock(d2, gf_dim / 2, 3, 1, res_net_block=res_net_block, name="D1")

    tensor = tf.keras.layers.Concatenate(3)([e0, d1])
    outputs = tf.keras.layers.Conv2D(c_dim, kernel_size=3, strides=1, padding="SAME",
                                     use_bias=True, name="output", activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_normal_initializer(0., 0.02))(tensor)

    inputs = [inputs_line, inputs_hint]
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.h0 = DisConvBlock(df_dim / 2, 4, 2)
        self.h1 = DisConvBlock(df_dim / 2, 3, 1)
        self.h2 = DisConvBlock(df_dim * 1, 4, 2)
        self.h3 = DisConvBlock(df_dim * 1, 3, 1)
        self.h4 = DisConvBlock(df_dim * 2, 4, 2)
        self.h5 = DisConvBlock(df_dim * 2, 3, 1)
        self.h6 = DisConvBlock(df_dim * 4, 4, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.last = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=tf.initializers.he_normal())

#     @tf.contrib.eager.defun
    def call(self, inputs, training):
        tensor = self.h0(inputs, training)
        tensor = self.h1(tensor, training)
        tensor = self.h2(tensor, training)
        tensor = self.h3(tensor, training)
        tensor = self.h4(tensor, training)
        tensor = self.h5(tensor, training)
        tensor = self.h6(tensor, training)
        tensor = self.flatten(tensor)  # (?,16384)
        tensor = self.last(tensor)
        return tensor