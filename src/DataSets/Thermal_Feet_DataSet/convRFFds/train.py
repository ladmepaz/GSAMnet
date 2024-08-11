import wandb
from wandb.integration.keras import WandbCallback
from loss.dice import DiceCoefficient
from Metrics import (jaccard,
                     sensitivity,
                     specificity,
                     )
from Metrics.dice import DiceCoefficientMetric
from .data import get_data
import tensorflow as tf
import numpy as np
import pandas as pd


def calculate_metrics_table(model, metrics, dataset_class, **kwargs_data_augmentation):

    *_, data = get_data(dataset_class, data_augmentation=False,
                        return_label_info=True,
                        **kwargs_data_augmentation)
    data = data.unbatch().batch(1)

    labels = getattr(dataset_class(), "labels_info", None)
    labels = list(labels.keys()) if labels else [None]

    results = {}

    for label in labels:
        if label != None:
            tem_data = data.filter(lambda x, y, l, *id_: l[0] == label)
        else:
            tem_data = data
        for metric in metrics:
            name_metric = metric.name
            curr = []
            for x, y, *_ in tem_data:
                y_pred = model(x)
                curr.append(metric.compute(y, y_pred))
            mean = abs(np.mean(curr))
            std = np.std(curr)
            results.setdefault('metric', []).append(name_metric)
            results.setdefault('label', []).append(label)
            results.setdefault('mean', []).append(mean)
            results.setdefault('std', []).append(std)

    return pd.DataFrame(results)


def get_compile_parameters():
    return {'loss': DiceCoefficient(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-3),
            'metrics': [jaccard(),
                        sensitivity(),
                        specificity(),
                        DiceCoefficientMetric()
                        ]
            }


def get_train_parameters(dataset_class, data_augmentation=True, validation=True, **kwargs_data_augmentation):
    train_data, val_data, test_data = get_data(
        dataset_class, data_augmentation=data_augmentation, **kwargs_data_augmentation)
    params = {'x': train_data,
              'validation_data': val_data,
              'epochs': 200,
              'callbacks': [WandbCallback(save_model=True)], }
    if not validation:
        params.pop('validation_data')
    return params


def train(model, dataset_class, run=None,
          data_augmentation=True,
          get_compile_parameters=get_compile_parameters,
          get_train_parameters=get_train_parameters,
          validation=True,
          **kwargs_data_augmentation):

    train_parameters = get_train_parameters(dataset_class, data_augmentation=data_augmentation,
                                            validation=validation,
                                            **kwargs_data_augmentation)
    compile_parameters = get_compile_parameters()
    metrics = get_compile_parameters()['metrics']
    model.compile(**compile_parameters)
    model.fit(**train_parameters)
    df_results = calculate_metrics_table(
        model, metrics, dataset_class, **kwargs_data_augmentation)
    if run:
        table = wandb.Table(dataframe=df_results)
        run.log({"metrics": table})
