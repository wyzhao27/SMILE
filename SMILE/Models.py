import os
from backbone import ResNeXt as base_model
import keras
import keras.ops as ops


class MILNetwork:
    def __init__(self, input_shape, ini_lr=1e-4):
        self.input_shape = input_shape
        self.ini_lr = ini_lr
        self.filter_list = [16, 32, 64, 128]
        self.conv_time_list = [2, 2, 4, 2]

    def Networks(self, flooding=0.0, name=None):
        input_layer = keras.layers.Input(shape=self.input_shape)

        time_num = self.input_shape[-2]
        instance_num = self.input_shape[-1]

        classify_encoder = ClassifyNetwork((self.input_shape[0], self.input_shape[1], self.input_shape[2], time_num),
                                           filter_list=self.filter_list, conv_time_list=self.conv_time_list).Networks("Classify_Encoder")

        instance_list = []

        for i in range(instance_num):
            instance_list.append(input_layer[:, :, :, :, :, i])

        instance_feature_list = []
        time_feature_among_instance_list = [[] for _ in range(time_num)]

        for i in range(instance_num):
            temp = classify_encoder(instance_list[i])
            instance_feature_list.append(temp)
            for k in range(time_num):
                time_feature_among_instance_list[k].append(temp[k])
        
        # multi-temporal-instance learning
        if time_num > 1:
            time_encoder = self.multi_time_per_scan(instance_feature_list[0], "Time_Encoder_For_Instance")
            time_feature_of_instance_list = []
            for i in range(instance_num):
                time_feature_of_instance_list.append(time_encoder(instance_feature_list[i]))
            
            time_feature_of_all_instance = ops.concatenate(time_feature_of_instance_list, axis=-1)
        else:
            time_feature_of_all_instance = ops.concatenate(instance_feature_list, axis=-1)
        
        feature_path1 = self.Res_Conv(time_feature_of_all_instance, conv_filter=self.filter_list[3], stride=1)

        # multi-instance temporal analysis
        if time_num > 1:
            instance_feature_of_time_list = []
            instance_encoder = self.multi_scan_per_time(time_feature_among_instance_list[0], "Instance_Encoder_Per_Time")
            for i in range(time_num):
                instance_feature_of_time_list.append(instance_encoder(time_feature_among_instance_list[i]))
            
            mil_time_feature_encoder = self.multi_time_per_scan(instance_feature_of_time_list, "MIL_Time_Feature_Encoder")
            feature_path2 = mil_time_feature_encoder(instance_feature_of_time_list)
            feature = ops.concatenate([feature_path1, feature_path2], axis=-1)
        else:
            feature = feature_path1

        y = self.Res_Conv(feature, conv_filter=self.filter_list[3], stride=1)

        y = keras.layers.GlobalAveragePooling3D()(y)

        y = keras.layers.Dense(1, activation='sigmoid')(y)

        model = keras.Model(inputs=input_layer, outputs=y, name=name)

        loss_func = keras.losses.BinaryFocalCrossentropy(label_smoothing=0.1, name="focal_loss", 
                                            reduction=None)
        loss_func_flooding = self.loss_with_flooding(loss_func, flooding_level=flooding)

        auc = keras.metrics.AUC(name='auc')

        model.compile(keras.optimizers.AdamW(weight_decay=self.ini_lr * 0.1, learning_rate=self.ini_lr), 
                      loss=loss_func_flooding,
                      metrics=[auc, loss_func])

        return model

    def loss_with_flooding(self, loss_func, flooding_level=0.02):
        def _main_func(y_true, y_pred):
            loss = loss_func(y_true, y_pred)
            loss = ops.abs(loss - flooding_level) + flooding_level
            return loss
        return _main_func

    def Res_Conv(self, input_layer, conv_filter, stride=1):
        res_path = keras.layers.Conv3D(conv_filter, 1, strides=stride, padding="same",
                          kernel_initializer="he_normal")(input_layer)
        res_path = keras.layers.GroupNormalization()(res_path)

        x = keras.layers.Conv3D(conv_filter*2, 3, strides=stride, padding="same", groups=self.filter_list[0],
                   kernel_initializer="he_normal")(input_layer)
        x = keras.layers.GroupNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv3D(conv_filter, 1, strides=1, padding="same",
                   kernel_initializer="he_normal")(x)
        x = keras.layers.GroupNormalization()(x)

        add_path = keras.layers.add([res_path, x])
        add_path = keras.layers.LeakyReLU()(add_path)

        return add_path

    def multi_time_per_scan(self, input_feature_list, name=None):
        tensor_shape = ops.shape(input_feature_list[0])

        input_list = []
        for i in range(len(input_feature_list)):
            input_list.append(keras.layers.Input(shape=tensor_shape[1:]))

        mha_atten = keras.layers.MultiHeadAttention(num_heads=8, key_dim=8)

        expan_list = []
        for item in input_list:
            reshape_item = keras.layers.Reshape((int(tensor_shape[1]*tensor_shape[2]*tensor_shape[3]), tensor_shape[4]))(item)
            expan_list.append(reshape_item)

        atten = mha_atten(expan_list[-1], ops.concatenate(expan_list[:-1], axis=1))
        y = expan_list[-1] - atten
        y = keras.layers.Reshape((tensor_shape[1], tensor_shape[2], tensor_shape[3], tensor_shape[4]))(y)
        y = self.Res_Conv(y, self.filter_list[3])

        model = keras.Model(inputs=input_list, outputs=y, name=name)
        return model

    def multi_scan_per_time(self, scan_list, name=None):
        input_list = []
        for i in range(len(scan_list)):
            input_list.append(keras.layers.Input(shape=ops.shape(scan_list[i])[1:]))

        feature = ops.concatenate(input_list, axis=-1)
        y = self.Res_Conv(feature, conv_filter=self.filter_list[3], stride=1)
        model = keras.Model(inputs=input_list, outputs=y, name=name)
        return model
    

class ClassifyNetwork:
    def __init__(self, input_shape, filter_list, conv_time_list):
        self.input_shape = input_shape
        self.filter_list = filter_list
        self.conv_time_list = conv_time_list

    def Networks(self, name=None):
        input_layer = keras.layers.Input(shape=self.input_shape)

        dim = self.input_shape[-1]

        Encoder = base_model(input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2], 1), 
                             filter_list=self.filter_list, conv_time_list=self.conv_time_list).Networks("Encoder")

        if dim == 3:
            feature_list_0 = Encoder(input_layer[:, :, :, :, 0:1])
            feature_list_1 = Encoder(input_layer[:, :, :, :, 1:2])
            feature_list_2 = Encoder(input_layer[:, :, :, :, 2:3])

            feature_0 = feature_list_0
            feature_1 = feature_list_1
            feature_2 = feature_list_2

            model = keras.Model(inputs=input_layer, outputs=[feature_0, feature_1, feature_2], name=name)

            return model
        if dim == 2:
            feature_list_0 = Encoder(input_layer[:, :, :, :, 0:1])
            feature_list_1 = Encoder(input_layer[:, :, :, :, 1:2])

            feature_0 = feature_list_0
            feature_1 = feature_list_1

            model = keras.Model(inputs=input_layer, outputs=[feature_0, feature_1], name=name)

            return model
        else:
            feature_list_0 = Encoder(input_layer)

            feature_0 = feature_list_0

            model = keras.Model(inputs=input_layer, outputs=feature_0, name=name)

            return model            

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    NetModel = MILNetwork((64, 64, 64, 3, 7))
    model = NetModel.Networks()
    model.summary()
