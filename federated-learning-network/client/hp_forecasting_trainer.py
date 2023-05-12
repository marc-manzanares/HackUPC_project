

class HPForecastingTrainer:
    def __init__(self, model_params, client_config):
        print('Initializing MnistModelTrainer...')
        self.ACCURACY_THRESHOLD = 0.5
        self.training_dataloader = None
        self.validation_dataloader = None
        self.client_config = client_config
        self.model_params = model_params

    def train_model(self):
        pass
        #return self.model_params