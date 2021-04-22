import tensorflow as tf
from backbones.backbone import Backbone
from backbones.resnet_common import Subsample
from backbones.resnet_common import bottleneck_v1


class ResNetV1(Backbone):
    BLOCKS = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self,
                 depth,
                 normalization,
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 frozen_normalization=False,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(ResNetV1, self).__init__(
            output_indices, strides, dilation_rates, frozen_stages, frozen_normalization, **kwargs
        )

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.depth = depth
        self.normalization = normalization

        self.blocks = ResNetV1.BLOCKS[depth]

        self.model = self.resnet()

    def composite_connect(self, previous, current):


    def resnet(self, input_shape=(224, 224, 3)):
        img_input = tf.keras.layers.Input(shape=input_shape)
        trainable = False if 1 in self.frozen_stages else True
        inputs = tf.keras.layers.Lambda(
            function=lambda inp: inp - tf.constant([123.68, 116.78, 103.94], dtype=self.dtype)
        )(img_input)
        x = tf.keras.layers.ZeroPadding2D(((3, 3), (3, 3)), name="conv1_pad")(inputs)
        x = tf.keras.layers.Conv2D(
            64, (7, 7), self.strides[0], "valid", dilation_rate=self.dilation_rates[0],
            use_bias=True, trainable=trainable, kernel_initializer="he_normal", name="conv1_conv"
        )(x)
        x = self.normalization(
            axis=-1, trainable=self.frozen_normalization and trainable, name="conv1_bn"
        )(x)
        x = tf.keras.layers.ReLU(name="conv1_relu")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
        x = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", name="pool1_pool")(x)

        block_outputs = [x]
        trainable = False if 2 in self.frozen_stages else True
        x = self.stack(x, 64, 1, self.dilation_rates[1], trainable, self.blocks[0], "conv2")
        block_outputs.append(x)
        trainable = False if 3 in self.frozen_stages else True
        x = self.stack(x, 128, self.strides[2], self.dilation_rates[2], trainable, self.blocks[1], "conv3")
        block_outputs.append(x)
        trainable = False if 4 in self.frozen_stages else True
        x = self.stack(x, 256, self.strides[3], self.dilation_rates[3], trainable, self.blocks[2], "conv4")
        block_outputs.append(x)
        trainable = False if 5 in self.frozen_stages else True
        x = self.stack(x, 512, self.strides[4], self.dilation_rates[4], trainable, self.blocks[3], "conv5")
        block_outputs.append(x)

        if -1 in self.output_indices:
            x = tf.keras.layers.GlobalAvgPool2D(name="avg_pool")(x)
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            outputs = tf.keras.layers.Dense(
                self.num_classes, activation="softmax", name="probs"
            )(x)
        else:
            outputs = [block_outputs[i] for i in self.output_indices]

        return tf.keras.Model(inputs=img_input, outputs=outputs, name="resnet" + str(self.depth))

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        x = bottleneck_v1(
            x, filters, strides, 1 if strides > 1 else dilation_rate,
            self.normalization, trainable, self.frozen_normalization, True, name + '_block1'
        )
        for i in range(2, blocks + 1):
            x = bottleneck_v1(
                x, filters, 1, dilation_rate, self.normalization, trainable,
                self.frozen_normalization, False, name=name + '_block' + str(i)
            )

        return x

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def init_weights(self, pre_trained_weights_path):
        self.model.summary()
        if pre_trained_weights_path is not None:
            self.model.load_weights(pre_trained_weights_path, by_name=True)
            tf.print("Initialized weights from", pre_trained_weights_path)
        else:
            tf.print(pre_trained_weights_path, "not exists! Initialized weights from scratch.")


if __name__ == '__main__':
    resnet = ResNetV1(50, tf.keras.layers.BatchNormalization, [-1], frozen_stages=[-1])
    # resnet.init_weights("/home/bail/workspace/pretrained_weights/resnet_v1_50.ckpt")
    resnet.init_weights("/home/bail/workspace/pretrained_weights/resnet50.h5")

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    cls = resnet(images, training=False)

    print(tf.argmax(tf.squeeze(cls, axis=0)))
    print(tf.reduce_max(cls))