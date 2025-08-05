import keras

class ResNeXt:
    def __init__(self, input_shape, filter_list=None, conv_time_list=None):
        if filter_list is None:
            filter_list = [32, 64, 128, 256]
        self.input_shape = input_shape
        self.filter_list = filter_list
        if conv_time_list is None:
            conv_time_list = [2, 2, 6, 2]
        self.conv_time_list = conv_time_list

    def Networks(self, name=None):
        input_layer = keras.layers.Input(shape=self.input_shape)
        y = self.Encoder_Path(input_layer, self.filter_list)

        model = keras.Model(inputs=input_layer, outputs=y, name=name)
        return model

    def Encoder_Path(self, input_layer, filter_list):
        y = input_layer

        for _ in range(self.conv_time_list[0]):
            y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[0], stride=1)

        y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[1], stride=2)

        for _ in range(self.conv_time_list[1]-1):
            y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[1], stride=1)

        y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[2], stride=2)

        for _ in range(self.conv_time_list[2]-1):
            y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[2], stride=1)

        y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[3], stride=2)

        for _ in range(self.conv_time_list[3]-1):
            y = self.Double_Res_Conv(input_layer=y, conv_filter=filter_list[3], stride=1)
        
        return y

    def Double_Res_Conv(self, input_layer, conv_filter, stride=1):
        res_path = keras.layers.Conv3D(conv_filter, 1, strides=stride, padding="same",
                          kernel_initializer="he_normal")(input_layer)
        res_path = keras.layers.GroupNormalization(groups=self.filter_list[0])(res_path)

        x = keras.layers.Conv3D(conv_filter, 1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input_layer)
        x = keras.layers.GroupNormalization(groups=self.filter_list[0])(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv3D(conv_filter*2, 3, strides=1, padding="same",groups=self.filter_list[0],
                   kernel_initializer="he_normal")(x)
        x = keras.layers.GroupNormalization(groups=self.filter_list[0]//2)(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv3D(conv_filter, 1, strides=1, padding="same",
                   kernel_initializer="he_normal")(x)
        x = keras.layers.GroupNormalization(groups=self.filter_list[0])(x)

        add_path = keras.layers.add([res_path, x])
        add_path = keras.layers.LeakyReLU()(add_path)

        return add_path
    

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

