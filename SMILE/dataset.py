import os, random
import SimpleITK as sitk
import numpy as np
import tensorflow as tf

class MyDataset:
    def __init__(self, true_list, false_list, true_path, false_path, batch_size, img_shape, spacing=0.75):
        self.true_list = true_list
        self.false_list = false_list
        self.true_path = true_path
        self.false_path = false_path
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.spacing = spacing

    def get_DataSet(self, mode="train"):
        true_dir_list = [self.true_path]  * len(self.true_list)
        true_y = [[1.] for _ in range(len(self.true_list))]
        false_dir_list = [self.false_path] * len(self.false_list)
        false_y = [[0.] for _ in range(len(self.false_list))]

        if mode == "train" or mode == "val":
            true_dataset = tf.data.Dataset.from_tensor_slices({"dir": true_dir_list, "img": self.true_list, "gt": true_y})

            true_dataset = true_dataset.repeat()
            true_dataset = true_dataset.shuffle(len(self.true_list))
            true_dataset = true_dataset.map(self.map_func(mode),
                                            num_parallel_calls=max(12, self.batch_size))
            true_dataset = true_dataset.prefetch(buffer_size=50)
            
            false_dataset = tf.data.Dataset.from_tensor_slices({"dir": false_dir_list, "img": self.false_list, "gt": false_y})
            false_dataset = false_dataset.repeat()
            false_dataset = false_dataset.shuffle(len(self.false_list))
            false_dataset = false_dataset.map(self.map_func(mode),
                                            num_parallel_calls=max(12, self.batch_size))
            false_dataset = false_dataset.prefetch(buffer_size=50)

            dataset = tf.data.Dataset.sample_from_datasets([true_dataset, false_dataset], [0.25, 0.75])
        else:
            dataset = tf.data.Dataset.from_tensor_slices({"dir": true_dir_list+false_dir_list, 
                                                          "img": self.true_list+self.false_list, 
                                                          "gt": true_y+false_y})

            dataset = dataset.map(self.map_func(mode),
                                            num_parallel_calls=max(12, self.batch_size))
            dataset = dataset.prefetch(buffer_size=50)
        
        dataset = dataset.batch(self.batch_size, num_parallel_calls=max(12, self.batch_size))
        dataset = dataset.prefetch(buffer_size=100)

        return dataset

    def map_func(self, mode):
        def map_function(item):
            x = tf.py_function(self.generate_func, inp=[item["dir"], item["img"], mode], Tout=tf.float32)
            x.set_shape(tf.TensorShape(self.img_shape))
            y = item["gt"]
            return x, y
        return map_function

    def generate_func(self, root_path, item, mode_byte):
        img_dir = str(root_path.numpy(), encoding="utf-8")
        file_name = str(item.numpy(), encoding="utf-8")
        mode = str(mode_byte.numpy(), encoding="utf-8")
        
        instance = self.img_shape[-1]
        instance_list = []

        for i in range(instance):
            instance_list.append(self.get_img_single_time(img_dir, file_name, str(i), mode))
        x = np.stack(instance_list, axis=-1)
        return tf.convert_to_tensor(x, dtype=tf.float32)

    def get_img_single_time(self, img_dir, file_name, index, mode):
        if self.img_shape[-2] == 3:
            x0 = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y0_"+file_name+".nii.gz"), mode)
            x1 = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y1_"+file_name+".nii.gz"), mode)
            x2 = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y2_"+file_name+".nii.gz"), mode)
            x = np.concatenate((x0, x1, x2), axis=-1)
        elif self.img_shape[-2] == 2:
            x1 = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y1_"+file_name+".nii.gz"), mode)
            x2 = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y2_"+file_name+".nii.gz"), mode)
            x = np.concatenate((x1, x2), axis=-1)
        elif self.img_shape[-2] == 1:
            x = self.data_generator(os.path.join(os.path.join(img_dir, file_name), index+"_y2_"+file_name+".nii.gz"), mode)
        else:
            raise ValueError("The time number is not 1, 2 or 3")
        return x

    def data_generator(self, input_file, mode):
        if os.path.exists(input_file):
            if mode == "train":
                max_angle = 90.
                max_translation = 0.
                angle = np.random.random(3) * max_angle / 180 * np.pi - 0.5 * max_angle / 180 * np.pi
                shift = np.random.random(3) * max_translation - 0.5 * max_translation
                zoom = np.random.uniform(1., 1., 1)
                flip = np.random.uniform(0., 1., 3)
                flip = np.where(flip > 0.5, True, False)

                inputImage = sitk.ReadImage(input_file)

                spacing = inputImage.GetSpacing()
                origin = inputImage.GetOrigin()
                direction = inputImage.GetDirection()
                size = inputImage.GetSize()
                if np.random.random() > 0.2:
                    imageCenterX = origin[0] + (spacing[0] * size[0] / 2.0) * direction[0]
                    imageCenterY = origin[1] + (spacing[1] * size[1] / 2.0) * direction[4]
                    imageCenterZ = origin[2] + (spacing[2] * size[2] / 2.0) * direction[8]

                    transform = sitk.Similarity3DTransform()
                    transform.SetCenter((imageCenterX, imageCenterY, imageCenterZ))
                    parameters = transform.GetParameters()
                    parameters = list(parameters)
                    parameters[0] = angle[0]
                    parameters[1] = angle[1]
                    parameters[2] = angle[2]
                    parameters[3] = shift[0] / spacing[0]
                    parameters[4] = shift[1] / spacing[1]
                    parameters[5] = shift[2] / spacing[2]
                    parameters[6] = zoom[0]
                    transform.SetParameters(parameters)

                    resample = sitk.ResampleImageFilter()
                    resample.SetInterpolator(sitk.sitkLinear)

                    resample.SetDefaultPixelValue(0)
                    resample.SetReferenceImage(inputImage)
                    resample.SetTransform(transform)
                    new = resample.Execute(inputImage)
                else:
                    new = inputImage

            elif mode == "val":
                new = sitk.ReadImage(input_file)
                flip = np.random.uniform(0., 1., 3)
                flip = np.where(flip > 0.5, True, False)
                size = new.GetSize()

            elif mode == "test":
                new = sitk.ReadImage(input_file)
                size = new.GetSize()

            else:
                raise ValueError("mode is not train or test")

            if size != self.img_shape[:-1]:
                new = new[size[0] // 2 - self.img_shape[0] // 2: size[0] // 2 + self.img_shape[0] // 2,
                          size[1] // 2 - self.img_shape[1] // 2: size[1] // 2 + self.img_shape[1] // 2,
                          size[2] // 2 - self.img_shape[2] // 2: size[2] // 2 + self.img_shape[2] // 2]

            if mode == "train" or mode == "val":
                flip_filter = sitk.FlipImageFilter()
                vector = tuple(flip.tolist())
                flip_filter.SetFlipAxes(vector)
                new = flip_filter.Execute(new)

            tmp = sitk.GetArrayFromImage(new)
        else:
            tmp = np.zeros((self.img_shape[0], self.img_shape[1], self.img_shape[2])) + 0.01
        tmp = np.expand_dims(tmp, axis=-1)
        return tmp
    
if __name__ == '__main__':
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

