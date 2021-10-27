from os.path import join

from config import Configuration

CONFIG_DIR = "settings"
CONFIG_NAME = "config"
CONFIG_EXTENSION = 'json'
CONFIG_FILENAME = ".".join((CONFIG_NAME, CONFIG_EXTENSION))
path_to_config = join(CONFIG_DIR, CONFIG_FILENAME)
config = Configuration(path_to_config)

data = config.get_data()
trainer = config.get_trainer()
fit_params = config.get_fit_params(data)
results = trainer.fit(**fit_params)

print(config.get_str_results(results))

graph_results = config.get_graph_results(results)
if graph_results is not None:
    graph_function, graph_params = graph_results
    graph_function(**graph_params)
