

class HPForecastingTrainer:
    def __init__(self, model_params, client_config):
        print('Initializing HPForecastModelTrainer...')
        self.ACCURACY_THRESHOLD = 0.5
        self.training_dataloader = None
        self.validation_dataloader = None
        self.client_config = client_config
        self.model_params = model_params
        #TODO Which are used and which useless?

    def train_model(self):
        #TODO Put the model calculations here
        print("hello world")
        model_params = 0
        return self.model_params

    def __load_datasets(self):
        #TODO get the dataset to work with
        print('Dataset ready to be used')
        pass