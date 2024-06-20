"""
===============================
Infrared Thermal Images of Feet
===============================
"""

import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


class InfraredThermalFeet:
    already_unzipped = False

    def __init__(self, split=[0.2, 0.2], seed: int = 42,
                 id_: str = "1HZa4pVwlIXCrRIidflB158kmtYGW23Qe"):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        self.split = listify(split)
        self.seed = seed

        self.__id = id_
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'InfraredThermalFeet')
        self.__path_images = os.path.join(self.__folder,
                                          'dataset' + "\\")

        if not InfraredThermalFeet.already_unzipped:
            self.__set_env()
            InfraredThermalFeet.already_unzipped = True

        self.file_images = glob(os.path.join(
            self.__path_images, '*[!(mask)].jpg'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images = sorted(self.file_images)

        self.groups = list(map(lambda x: os.path.split(x)[-1].split('_')[0],
                               self.file_images))

        self.num_samples = len(self.file_images)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'InfraredThermalFeet.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    @staticmethod
    def __preprocessing_mask(mask):
        mask = mask[..., 0] > 0.5
        mask = mask.astype(np.float32)
        return mask[..., None]

    def load_instance_by_id(self, id_img):
        root_name = os.path.join(self.__path_images, id_img)
        return self.load_instance(root_name)

    @staticmethod
    def load_instance(root_name):
        img = cv2.imread(f'{root_name}.jpg')/255
        img = img[..., 0][..., None]
        mask = cv2.imread(f'{root_name}_mask.png')
        mask = InfraredThermalFeet.__preprocessing_mask(mask)
        id_image = os.path.split(root_name)[-1]
        return img, mask, id_image

    @staticmethod
    def __gen_dataset(file_images):
        def generator():
            for root_name in file_images:
                yield InfraredThermalFeet.load_instance(root_name)
        return generator

    def __generate_tf_data(self, files):
        output_signature = (tf.TensorSpec((None, None, None), tf.float32),
                            tf.TensorSpec((None, None, None), tf.float32),
                            tf.TensorSpec(None, tf.string),
                            )

        dataset = tf.data.Dataset.from_generator(self.__gen_dataset(files),
                                                 output_signature=output_signature)

        len_files = len(files)
        dataset = dataset.apply(
            tf.data.experimental.assert_cardinality(len_files))
        return dataset

    def __get_log_tf_data(self, i, files):
        print(f' Number of images for Partition {i}: {len(files)}')
        return self.__generate_tf_data(files)

    @staticmethod
    def __shuffle(X, seed):
        np.random.seed(seed)
        np.random.shuffle(X)

    @staticmethod
    def _train_test_split(X, groups, random_state=42, test_size=0.2):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        indxs_train, index_test = next(gss.split(X, groups=groups))

        InfraredThermalFeet.__shuffle(indxs_train, random_state)
        InfraredThermalFeet.__shuffle(index_test, random_state)

        return X[indxs_train], X[index_test], groups[indxs_train], groups[index_test]

    def __call__(self,):
        file_images = np.array(self.file_images)
        groups = np.array(self.groups)
        train_imgs, test_imgs, g_train, _ = InfraredThermalFeet._train_test_split(
            file_images,
            groups,
            test_size=self.split[0],
            random_state=self.seed
        )

        train_imgs, val_imgs, _, _ = InfraredThermalFeet._train_test_split(
            train_imgs, g_train,
            test_size=self.split[1],
            random_state=self.seed)

        p_files = [train_imgs, val_imgs, test_imgs]

        partitions = [self.__get_log_tf_data(
            i+1, p) for i, p in enumerate(p_files)]

        return partitions


class ModifiedInfraredThermalFeet(InfraredThermalFeet):
    def __init__(self, split=[0.2, 0.2], seed: int = 42,
                 file_path: str = "InfraredThermalFeet.zip",
                 *args, **kwargs):

        self.split = listify(split)
        self.seed = seed

        self.file_path = file_path
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'InfraredThermalFeet')
        self.__path_images = os.path.join(self.__folder,
                                          'dataset' + "\\")

        if not InfraredThermalFeet.already_unzipped:
            self.__set_env()
            InfraredThermalFeet.already_unzipped = True

        self.file_images = glob(os.path.join(
            self.__path_images, '*[!(mask)].jpg'))
        self.file_images = list(map(lambda x: x[:-4], self.file_images))
        self.file_images = sorted(self.file_images)

        self.groups = list(map(lambda x: os.path.split(x)[-1].split('_')[0],
                               self.file_images))
        self.num_samples = len(self.file_images)

    def __set_env(self):
        path_zip = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataSet',
                                'InfraredThermalFeet.zip')
        os.makedirs(self.__folder, exist_ok=True)
        unzip(path_zip, self.__folder)


if __name__ == "__main__":
    import convRFFds.data as data
    from utils import download_from_drive
    from utils import unzip
    from utils import listify
    kwargs_data_augmentation = dict(repeat=1,
                                    batch_size=2,
                                    shape=224,
                                    split=[0.4, 0.3]
                                    )
    dataset = ModifiedInfraredThermalFeet
    train_dataset, val_dataset, test_dataset = data.get_data(
        dataset_class=dataset, data_augmentation=False, return_label_info=True, **kwargs_data_augmentation)
else:
    from .utils import download_from_drive
    from .utils import unzip
    from .utils import listify
