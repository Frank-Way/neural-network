from os.path import join

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from networks import NeuralNetwork
from losses import MeanSquaredError, SoftmaxCrossEntropy
from optimizers import SGD, SGDMomentum
from trainers import Trainer
from layers import Dense
from operations import Sigmoid, Linear, LeakyReLU, ReLU, Tanh

from utils import show_results, load_config, calc_accuracy_model, show_results3d
from data_loaders import MNISTDataLoader, ApproximationDataLoader

layer_classes = {'dense': Dense}
activation_classes = {'sigmoid': Sigmoid, 'linear': Linear, 'leakyrelu': LeakyReLU, 'relu': ReLU, 'tanh': Tanh}
loss_classes = {'mse': MeanSquaredError, 'sce': SoftmaxCrossEntropy}
optimizer_classes = {'sgd': SGD, 'sgdm': SGDMomentum}
trainer_classes = {'t': Trainer}
data_loader_classes = {'mnist': MNISTDataLoader,
                       'ap': ApproximationDataLoader}

CONFIG_DIR = "settings"
CONFIG_NAME = "config"
CONFIG_EXTENSION = 'json'
CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)
config = load_config(path_to_config)

samples_dir_name = config["samples_dir_name"]
inputs = int(samples_dir_name.split("/")[-1])
sample_prefix = config["sample_prefix"]
size = config["size"]
batch_size = config["batch_size"]
epochs = config["epochs"]
query_every = epochs // config["query_times"]
layers = config["layers"]
early_stopping = config["early_stopping"]
neurons = config["neurons"]
lr = config["lr"]
final_lr = config["final_lr"]
dropout = config["dropout"]
weight_init = config["weight_init"]
decay_type = config["decay_type"]
momentum = config["momentum"]
layers_class = [layer_classes[lc] for lc in config["layers_class"]]
activations_class = [activation_classes[ac] for ac in config["activations_class"]]
loss_class = loss_classes[config["loss_class"]]
optimizer_class = optimizer_classes[config["optimizer_class"]]
trainer_class = trainer_classes[config["trainer_class"]]
data_loader_class = data_loader_classes[config["data_loader_class"]]
seed = config["seed"]
use_seed = config["use_seed"]
extension = config["sample_extension"]
scale_inputs = config["scale_inputs"]
print_results = config["print_results"]
limits = config["limits"]
show_plots = config["show_plots"]

if config["data_loader_class"] == "ap":
    dl = data_loader_class(join(samples_dir_name, f"{sample_prefix}_{size}.{extension}"),
                           scale_inputs=scale_inputs,
                           limits=limits)
elif config["data_loader_class"] == "mnist":
    dl = data_loader_class(join(samples_dir_name, "mnist"))
x_train, x_test, y_train, y_test = dl.load()

nn = NeuralNetwork(
    layers=[layers_class[ll](neurons=neurons[ll],
                             activation=activations_class[ll](),
                             dropout=dropout[ll],
                             weight_init=weight_init[ll])
            for ll in range(layers)],
    loss=loss_class()
)

if config["optimizer_class"] == "sgd":
    optimizer = optimizer_class(lr=lr,
                                final_lr=final_lr,
                                decay_type=decay_type)
elif config["optimizer_class"] == "sgdm":
    optimizer = optimizer_class(lr=lr,
                                final_lr=final_lr,
                                decay_type=decay_type,
                                momentum=momentum)

trainer = trainer_class(nn, optimizer)

(_loss, delta, _losses,
 _x_test, _y_test, _test_preds) = trainer.fit(x_train, y_train,
                                              x_test, y_test,
                                              epochs=epochs,
                                              query_every=query_every,
                                              batch_size=batch_size,
                                              early_stopping=early_stopping,
                                              seed=seed if use_seed else None,
                                              print_results=print_results)

# calc_accuracy_model(nn, x_test, y_test)

print(f"{neurons[:-1]}: {sample_prefix}_{size} - {delta}")
print(delta / np.abs(np.max(_y_test) - np.min(_y_test)) * 100, '%')
print(np.average(np.abs(_y_test - _test_preds)))

if show_plots:
    if inputs == 1:
        show_results(_losses,
                     _x_test,
                     _test_preds,
                     _y_test,
                     f"{sample_prefix}_{size}",
                     neurons)
    elif inputs == 2:
        show_results3d(_losses,
                       _x_test,
                       _test_preds,
                       _y_test,
                       f"{sample_prefix}_{size}",
                       neurons)
    else:
        print(f"Plotting is unavailable for {inputs}-D inputs")
