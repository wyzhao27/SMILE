import multiprocessing as mp
import os, argparse, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Models import MILNetwork as Network
from dataset import MyDataset
import warnings
warnings.filterwarnings("ignore")
from absl import logging
logging.set_verbosity(logging.ERROR)

base_img_shape = (64, 64, 64)
true_subset = ""
false_subset = ""
true_path = ""
false_path = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=["0"], nargs="+", type=str, required=True)
    parser.add_argument("--subset", default=["0"], nargs="+", type=str, required=True)
    parser.add_argument("--batch", default=2, type=int, required=False)
    parser.add_argument("--lr", default=1e-4, type=float, required=False)
    parser.add_argument("--verbose", default=0, type=int, required=False)
    parser.add_argument("--epoch", default=150, type=int, required=False)
    parser.add_argument("--dim", default=3, type=int, required=False)
    parser.add_argument("--instance", default=3, type=int, required=False)
    parser.add_argument("--flooding", default=0.0, type=float, required=False)
    parser.add_argument("--parallel", default=0.0, type=float, required=False)
    parser.add_argument("--epochs_drop", default=50.0, type=float, required=False)
    args = parser.parse_args()
    return args


def get_list_from_pool(subset_path, train_pool, val_pool):
    npz = np.load(subset_path)

    train_list = []
    val_list = []

    for item in train_pool:
        train_list = train_list + npz["subset" + item].tolist()

    for item in val_pool:
        val_list = val_list + npz["subset" + item].tolist()

    return train_list, val_list


def train_func(args_item, subset, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    weight_hdf = "smile_instance_" + str(args_item.instance) + "_fold" + subset + ".weights.h5"

    img_shape = (base_img_shape[0], base_img_shape[1], base_img_shape[2], args_item.dim, args_item.instance)
    
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        NetModel = Network(img_shape, ini_lr=args_item.lr)
        model = NetModel.Networks(flooding=args_item.flooding)

    if os.path.exists(weight_hdf):
        model.load_weights(weight_hdf)
        print("Subset " + subset + ": Weights loaded...")

    print("Subset " + subset + ": Loading Data...")
    num = int(subset)

    train_pool = []
    val_pool = [str((num+1) % 5)] 
    test_pool = [str(num)]

    for i in range(3):
        train_pool.append(str((num + 2 + i) % 5))

    print(test_pool, val_pool, train_pool)

    true_train_list, true_val_list = get_list_from_pool(true_subset, train_pool, val_pool)
    false_train_list, false_val_list = get_list_from_pool(false_subset, train_pool, val_pool)

    batch_size = args_item.batch

    train_set = MyDataset(true_list=true_train_list, false_list=false_train_list, true_path=true_path,
                          false_path=false_path, batch_size=batch_size, img_shape=img_shape).get_DataSet("train")

    val_set = MyDataset(true_list=true_val_list, false_list=false_val_list, true_path=true_path, false_path=false_path,
                        batch_size=batch_size, img_shape=img_shape).get_DataSet("val")

    train_set = strategy.experimental_distribute_dataset(train_set)
    val_set = strategy.experimental_distribute_dataset(val_set)

    train_step = (len(true_train_list) + len(false_train_list)) // batch_size
    val_step = (len(true_val_list) + len(false_val_list)) // batch_size

    baseline = 100.
    if os.path.exists(weight_hdf):
        initial_metric = model.evaluate(val_set, steps=val_step, verbose=0, return_dict=True)
        baseline = initial_metric["focal_loss"]
        print("Subset " + subset + ": Initial loss: " + str(baseline))

    print("Subset " + subset + ": Fitting model...")
    model_checkpoint = ModelCheckpoint(weight_hdf, monitor='val_focal_loss', save_best_only=True, mode='min', verbose=1,
                                       initial_value_threshold=baseline, save_weights_only=True)

    def step_decay(epoch):
        initial_lrate = args_item.lr
        drop = 0.2
        epochs_drop = args_item.epochs_drop
        lrate = max(initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop)),5e-6)
        return lrate

    reduce_lr = LearningRateScheduler(step_decay)

    model.fit(x=train_set, steps_per_epoch=train_step, epochs=args_item.epoch, verbose=args_item.verbose,
              validation_data=val_set, validation_steps=val_step,
              callbacks=[model_checkpoint, reduce_lr])

    print("Subset " + subset + ": Training completed...")

if __name__ == '__main__':
    args = parse_args()

    parallel_mode = args.parallel

    if parallel_mode > 0.5:
        pid_list = []
        for i in range(len(args.subset)):
            p = mp.Process(target=train_func,
                           args=(args, args.subset[i], args.gpu[i]))
            pid_list.append(p)
            p.start()

        for p in pid_list:
            p.join()

    else:
        pid_list = []
        for i in range(len(args.subset)):
            p = mp.Process(target=train_func,
                           args=(args, args.subset[i], args.gpu[i]))
            pid_list.append(p)
            p.start()
            p.join()



