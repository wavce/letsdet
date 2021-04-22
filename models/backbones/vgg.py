import os
import time
import tensorflow as tf
from models.builder import BACKBONES
from models.backbones import Backbone
from core.layers import build_convolution


class VGG(Backbone):
    BLOCKS = {
        16: [2, 2, 3, 3, 3],
        19: [2, 2, 4, 4, 4]
    }

    def __init__(self,
                 depth=16,
                 convolution="conv2d",
                 activation="relu",
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1, ),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(VGG, self).__init__(convolution=convolution,
                                  activation=activation,
                                  normalization=None,
                                  output_indices=output_indices,
                                  strides=strides,
                                  dilation_rates=dilation_rates,
                                  frozen_stages=frozen_stages,
                                  input_shape=input_shape,
                                  input_tensor=input_tensor,
                                  num_classes=num_classes,
                                  drop_rate=drop_rate,
                                  **kwargs)

        self._r_mean = 123.68
        self._g_mean = 116.78
        self._b_mean = 103.94

        self.depth = depth
        self.blocks = VGG.BLOCKS[depth]
        self.freeze_stages()

    def freeze_stages(self):
        if -1 in self.frozen_stages:
            return

        for block in self.frozen_stages:
            assert block >= 1, "The frozen stage must start from 1."
            base_name = "conv%d" % block
            layer = getattr(self, base_name)
            layer.trainable = False

    def build_model(self):
        outputs = []
        x = tf.keras.layers.Lambda(function=lambda inp: inp - [self._r_mean, self._b_mean, self._g_mean])(self.img_input)
        x = tf.keras.Sequential([
            build_convolution(self.convolution,
                              filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              dilation_rate=self.dilation_rates[0],
                              activation=self.activation,
                              name="conv1_%d" % (i+1))
            for i in range(VGG.BLOCKS[self.depth][0])], name="conv1")(x)
        x= tf.keras.layers.MaxPool2D((2, 2), self.strides[0], "same", name="pool1")(x)

        x = tf.keras.Sequential([
            build_convolution(self.convolution,
                              filters=128,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              dilation_rate=self.dilation_rates[1],
                              activation=self.activation,
                              name="conv2_%d" % (i+1))
            for i in range(VGG.BLOCKS[self.depth][1])], name="conv2")(x)
        outputs.append(x)
        x= tf.keras.layers.MaxPool2D((2, 2), self.strides[1], "same", name="pool2")(x)

        x = tf.keras.Sequential([
            build_convolution(self.convolution,
                              filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              dilation_rate=self.dilation_rates[2],
                              activation=self.activation,
                              name="conv3_%d" % (i+1))
            for i in range(VGG.BLOCKS[self.depth][2])], name="conv3")(x)
        outputs.append(x)
        x = tf.keras.layers.MaxPool2D((2, 2), self.strides[2], "same", name="pool3")(x)

        x = tf.keras.Sequential([
            build_convolution(self.convolution,
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              dilation_rate=self.dilation_rates[2],
                              activation=self.activation,
                              name="conv4_%d" % (i + 1))
            for i in range(VGG.BLOCKS[self.depth][3])], name="conv4")(x)
        outputs.append(x)
        x = tf.keras.layers.MaxPool2D((2, 2), self.strides[3], "same", name="pool4")(x)

        x = tf.keras.Sequential([
            build_convolution(self.convolution,
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              dilation_rate=self.dilation_rates[2],
                              activation=self.activation,
                              name="conv5_%d" % (i + 1))
            for i in range(VGG.BLOCKS[self.depth][4])], name="conv5")(x)
        outputs.append(x)

        if -1 not in self.output_indices:
            return (outputs[i-1] for i in self.output_indices)

        
        x = tf.keras.layers.MaxPool2D((2, 2), self.strides[4], "same", name="pool5")(x)
        x = build_convolution(self.convolution,
                                filters=4096,
                                kernel_size=(7, 7),
                                strides=(1, 1),
                                padding="valid",
                                activation=self.activation,
                                name="fc6")(x)
        x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
        x = build_convolution(self.convolution,
                                filters=4096,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding="same",
                                activation=self.activation,
                                name="fc7")(x)
        x = tf.keras.layers.Lambda(
            function=lambda inp: tf.reduce_mean(inp, [1, 2], keepdims=True),
            name="global_pool")(x)

        x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
        x = build_convolution(self.convolution,
                                filters=self.num_classes,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding="same",
                                kernel_initializer="he_normal",
                                name="fc8")(x)
        x = tf.keras.layers.Lambda(function=lambda inp: tf.squeeze(inp, axis=[1, 2]), name="squeeze")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name) 

    def init_weights(self, pre_trained_weights_path):
        if os.path.exists(pre_trained_weights_path):
            for weight in self.model.weights:
                name = weight.name
                name = name.split(":")[0]
                name = "vgg_" + str(self.depth) + "/" + name
                if "kernel" in name:
                    name = name.replace("kernel", "weights")
                if "bias" in name:
                    name = name.replace("bias", "biases")

                weight.assign(tf.train.load_variable(pre_trained_weights_path, name))

            tf.print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                     "Initialized weights from", pre_trained_weights_path)
        else:
            tf.print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                     pre_trained_weights_path, "not exist! Initialized weights from scratch.")


@BACKBONES.register
class VGG16(VGG):
    def __init__(self,
                 convolution="conv2d",
                 activation="relu",
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1, ),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(VGG16, self).__init__(depth=16,
                                    convolution=convolution,
                                    activation=activation,
                                    normalization=None,
                                    output_indices=output_indices,
                                    strides=strides,
                                    dilation_rates=dilation_rates,
                                    frozen_stages=frozen_stages,
                                    input_shape=input_shape,
                                    input_tensor=input_tensor,
                                    num_classes=num_classes,
                                    drop_rate=drop_rate,
                                    **kwargs)


@BACKBONES.register
class VGG19(VGG):
    def __init__(self,
                 convolution="conv2d",
                 activation="relu",
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1, ),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(VGG19, self).__init__(depth=19,
                                    convolution=convolution,
                                    activation=activation,
                                    normalization=None,
                                    output_indices=output_indices,
                                    strides=strides,
                                    dilation_rates=dilation_rates,
                                    frozen_stages=frozen_stages,
                                    input_shape=input_shape,
                                    input_tensor=input_tensor,
                                    num_classes=num_classes,
                                    drop_rate=drop_rate,
                                    **kwargs)


if __name__ == '__main__':
    vgg = VGG16([-1])
    vgg.init_weights("/home/bail/workspace/pretrained_weights/vgg_16.ckpt")

    with tf.io.gfile.GFile("/home/bail/workspace/face_detections/data/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    cls = vgg.model(images, training=False)

    print(tf.argmax(tf.squeeze(cls)))

