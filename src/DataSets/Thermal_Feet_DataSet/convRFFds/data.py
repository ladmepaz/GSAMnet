import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def resize(shape=(256, 256)):
    def func(img, mask):
        return tf.image.resize(img, shape), tf.image.resize(mask, shape)
    return func


def random_translation(img, mask, translation_h_w):
    shape_img = tf.cast(tf.shape(img), tf.float32)
    dx = tf.cast(shape_img[-2]*translation_h_w[1], tf.int32)
    dy = tf.cast(shape_img[-3]*translation_h_w[0], tf.int32)
    dx = tf.random.uniform(shape=(), minval=-dx, maxval=dx, dtype=tf.int32)
    dy = tf.random.uniform(shape=(), minval=-dy, maxval=dy, dtype=tf.int32)
    img = tfa.image.translate(img, [dx, dy], fill_mode='nearest')
    mask = tfa.image.translate(mask, [dx, dy], fill_mode='constant')
    return img, mask


def random_zoom(img, mask, zoom_h_w):
    img_shape = tf.shape(img)[-3:-1]

    h = tf.cast(img_shape[0], tf.float32)
    h = tf.cast(h*zoom_h_w[0], tf.int32)
    h = tf.random.uniform(shape=(), minval=-h, maxval=h, dtype=tf.int32)

    w = tf.cast(img_shape[1], tf.float32)
    w = tf.cast(w*zoom_h_w[1], tf.int32)
    w = tf.random.uniform(shape=(), minval=-w, maxval=w, dtype=tf.int32)

    img = tf.image.resize_with_crop_or_pad(
        img, img_shape[0]+h,  img_shape[1]+w)
    mask = tf.image.resize_with_crop_or_pad(
        mask, img_shape[0]+h,  img_shape[1]+w)

    img = tf.image.resize(img, img_shape)
    mask = tf.image.resize(mask, img_shape)

    return img, mask


def data_augmentation_func(flip_left_right=True,
                           flip_up_down=True, range_rotate=(-10, 10),
                           translation_h_w=None, zoom_h_w=None):
    def data_aug(img, mask):
        seed = tf.random.uniform(shape=(2,), minval=1,
                                 maxval=1000, dtype=tf.int32)

        if flip_left_right:
            seed = tf.random.uniform(
                shape=(2,), minval=1, maxval=1000, dtype=tf.int32)
            img = tf.image.stateless_random_flip_left_right(img, seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed)

        if flip_up_down:
            seed = tf.random.uniform(
                shape=(2,), minval=1, maxval=1000, dtype=tf.int32)
            img = tf.image.stateless_random_flip_up_down(img, seed)
            mask = tf.image.stateless_random_flip_up_down(mask, seed)

        if range_rotate:
            min_val = range_rotate[0]/180*np.pi
            max_val = range_rotate[1]/180*np.pi
            rotation = tf.random.uniform(
                shape=(), minval=min_val, maxval=max_val, dtype=tf.float32)
            img = tfa.image.rotate(img, rotation, fill_mode='reflect')
            mask = tfa.image.rotate(mask, rotation, fill_mode='constant')

        if translation_h_w:
            img, mask = random_translation(img, mask, translation_h_w)

        if zoom_h_w:
            img, mask = random_zoom(img, mask, zoom_h_w)

        return img, mask

    return data_aug


def preprocess_data(data, data_augmentation=False,
                    return_label_info=False, shape=256, repeat=1,
                    flip_left_right=True,
                    flip_up_down=True,
                    range_rotate=(-10, 10),
                    translation_h_w=None,
                    zoom_h_w=None):

    if not return_label_info:
        data = data.map(lambda *items: items[:2],
                        num_parallel_calls=tf.data.AUTOTUNE)

    data = data.map(lambda x, y, *l: (*resize((shape, shape))(x, y), *l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache()

    data = data.repeat(repeat)
    if data_augmentation:
        data = data.map(
            data_augmentation_func(
                flip_left_right=flip_left_right,
                flip_up_down=flip_up_down,
                range_rotate=range_rotate,
                translation_h_w=translation_h_w,
                zoom_h_w=zoom_h_w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return data


def get_data(dataset_class, seed=42,
             data_augmentation=True,
             return_label_info=False,
             shape=256, batch_size=32,
             repeat=1,
             flip_left_right=True,
             flip_up_down=True,
             range_rotate=(-10, 10),
             translation_h_w=None,
             zoom_h_w=None,
             split=None):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    if split:
        dataset = dataset_class(seed=seed, split=split)
    else:
        dataset = dataset_class(seed=seed)
    train_data, val_data, test_data = dataset()

    train_data = preprocess_data(
        train_data, data_augmentation,
        return_label_info, shape, repeat=repeat,
        flip_left_right=flip_left_right,
        flip_up_down=flip_up_down,
        range_rotate=range_rotate,
        translation_h_w=translation_h_w,
        zoom_h_w=zoom_h_w,
    ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    val_data = preprocess_data(
        val_data,
        data_augmentation=False,
        return_label_info=return_label_info,
        shape=shape,
    ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    test_data = preprocess_data(
        test_data,
        data_augmentation=False,
        return_label_info=return_label_info,
        shape=shape,
    ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)
    test_data = test_data.with_options(options)

    return train_data, val_data, test_data
