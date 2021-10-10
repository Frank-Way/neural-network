from os.path import join

from networks import NeuralNetwork
from losses import MeanSquaredError, SoftmaxCrossEntropy
from optimizers import SGD, SGDMomentum
from trainers import Trainer
from layers import Dense
from operations import Sigmoid, Linear, LeakyReLU, ReLU, Tanh

from utils import load_config
from data_loaders import MNISTDataLoader, ApproximationDataLoader


layer_classes = {'dense': Dense}
activation_classes = {'sigmoid': Sigmoid, 'linear': Linear, 'leakyrelu': LeakyReLU, 'relu': ReLU, 'tanh': Tanh}
loss_classes = {'mse': MeanSquaredError, 'sce': SoftmaxCrossEntropy}
optimizer_classes = {'sgd': SGD, 'sgdm': SGDMomentum}
trainer_classes = {'t': Trainer}
data_loader_classes = {'mnist': MNISTDataLoader,
                       'ap1': ApproximationDataLoader}

CONFIG_DIR = "../../settings"
CONFIG_NAME = "config"
CONFIG_EXTENSION = 'json'
CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)
config = load_config(path_to_config)

samples_dir_name = config["samples_dir_name"]
sample_prefix = config["sample_prefix"]
size = config["size"]
batch_size = config["batch_size"]
query_every = config["query_every"]
epochs = config["epochs"]
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

nns = [[8, 1], [16, 1], [32, 1]]
for neurons in nns:
    filenames = ["tan(pi*x:2)", "x^(1:2)", "sin(pi*x)", "2^-x", "ln(x)", "e^(-x)"]
    for filename in filenames:
        sizes = [512, 1024]
        for size in sizes:
            dl = data_loader_class(join(samples_dir_name, f"{filename}" + (f"_{size}" if size > 0 else "") + ".pkl"),
                                   scale_inputs=scale_inputs)
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
            min_delta = float("inf")
            min_results = None
            for e in [5000, 15000, 50000]:
                (_loss, delta, _losses,
                 _x_test, _y_test, _test_preds) = trainer.fit(x_train, y_train,
                                                              x_test, y_test,
                                                              epochs=e,
                                                              query_every=e//5,
                                                              batch_size=batch_size,
                                                              early_stopping=early_stopping,
                                                              seed=seed if use_seed else None,
                                                              print_results=False)

                print(f"{e}; {neurons[:-1]}: {filename}_{size} - {delta}")
            # print(delta / np.abs(np.max(_y_test) - np.min(_y_test)) * 100, '%')
            # show_results(_losses,
            #              _x_test,
            #              _test_preds,
            #              _y_test,
            #              f"{sample_prefix}_{size}",
            #              neurons)
