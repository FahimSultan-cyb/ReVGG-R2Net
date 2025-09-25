import tensorflow as tf
from tensorflow.keras import layers, models
from config import Config

class ReVGG_R2Net:
    def __init__(self):
        self.config = Config()
    
    def recurrent_conv_block(self, input_tensor, filters, t=None):
        if t is None:
            t = self.config.RECURRENT_STEPS
            
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization()(x)
        x1 = layers.Activation('relu')(x)
        
        for i in range(t):
            if i == 0:
                x2 = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
            else:
                x2 = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(layers.add([x1, x2]))
            x2 = layers.BatchNormalization()(x2)
            x2 = layers.Activation('relu')(x2)
        
        return layers.add([x1, x2])

    def residual_recurrent_conv_block(self, input_tensor, filters, t=None):
        if t is None:
            t = self.config.RECURRENT_STEPS
            
        if input_tensor.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor
        
        x = self.recurrent_conv_block(input_tensor, filters, t)
        return layers.add([shortcut, x])

    def upconv_block(self, input_tensor, skip_tensor, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        x = layers.concatenate([x, skip_tensor])
        x = self.residual_recurrent_conv_block(x, filters)
        return x

    def create_model(self):
        input_shape = (self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.INPUT_CHANNELS)
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        
        for layer in base_model.layers:
            layer.trainable = True
        
        e1 = base_model.get_layer("block1_conv2").output
        e2 = base_model.get_layer("block2_conv2").output  
        e3 = base_model.get_layer("block3_conv3").output
        e4 = base_model.get_layer("block4_conv3").output
        bridge = base_model.get_layer("block5_conv3").output
        
        e1 = self.residual_recurrent_conv_block(e1, 64)
        e2 = self.residual_recurrent_conv_block(e2, 128)
        e3 = self.residual_recurrent_conv_block(e3, 256)
        e4 = self.residual_recurrent_conv_block(e4, 512)
        bridge = self.residual_recurrent_conv_block(bridge, 1024)
        
        d4 = self.upconv_block(bridge, e4, 512)
        d3 = self.upconv_block(d4, e3, 256)
        d2 = self.upconv_block(d3, e2, 128)
        d1 = self.upconv_block(d2, e1, 64)
        
        outputs = layers.Conv2D(self.config.OUTPUT_CHANNELS, (1, 1), activation='sigmoid')(d1)
        
        model = models.Model(inputs=base_model.input, outputs=outputs)
        return model

def create_revgg_r2net():
    revgg = ReVGG_R2Net()
    return revgg.create_model()
