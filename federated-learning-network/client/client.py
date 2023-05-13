import json
import sys
import requests

from os import environ

from requests.exceptions import Timeout

from .utils import model_params_to_request_params
from .mnist_model_trainer import MnistModelTrainer
from .chest_x_ray_model_trainer import ChestXRayModelTrainer
from .hp_forecasting_trainer import HPForecastingTrainer
from .client_status import ClientStatus
from .config import DEFAULT_SERVER_URL
from .training_type import TrainingType

from web3 import Web3


class Client:
    def __init__(self, client_url):
        self.client_url = client_url
        self.status = ClientStatus.IDLE
        self.training_type = None
        self.SERVER_URL = environ.get('SERVER_URL')
        if self.SERVER_URL is None:
            print('Warning: SERVER_URL environment variable is not defined, using DEFAULT_SERVER_URL:',
                  DEFAULT_SERVER_URL)
            self.SERVER_URL = DEFAULT_SERVER_URL
        else:
            print('Central node URL:', self.SERVER_URL)

        if self.client_url is None:
            print('Error: client_url is missing, cannot create a client')
            return
        self.register()

    def do_training(self, training_type, model_params, federated_learning_config):
        if self.can_do_training():
            self.training_type = training_type
            print(federated_learning_config)

            if self.training_type == TrainingType.MNIST:
                client_model_trainer = MnistModelTrainer(model_params, federated_learning_config)
            elif self.training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
                client_model_trainer = ChestXRayModelTrainer(model_params, federated_learning_config, self.client_url)
            elif self.training_type == TrainingType.HP_FORECASTING_MODEL:
                client_model_trainer = HPForecastingTrainer(model_params, federated_learning_config)
            else:
                raise ValueError('Unsupported training type', training_type)

            self.status = ClientStatus.TRAINING
            print('Training started...')
            try:
                model_params_updated = client_model_trainer.train_model()
                model_params_updated = model_params_to_request_params(training_type, model_params_updated)
                self.update_model_params_on_server(model_params_updated)
            except Exception as e:
                raise e
            finally:
                self.status = ClientStatus.IDLE
                print('Training finished...')
        else:
            print('Training requested but client status is', self.status)
        sys.stdout.flush()

    def update_model_params_on_server(self, model_params):
        request_url = self.SERVER_URL + '/model_params'
        request_body = model_params
        request_body['client_url'] = self.client_url
        request_body['training_type'] = self.training_type
        print('Sending calculated model weights to central node')

        # Connect to the Ethereum network (replace "your_network_url" with the actual network URL)
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:9545/"))

        try:
            block_number = w3.eth.block_number
            print("Connected to the blockchain network")
        except:
            print("Not connected to the blockchain network")


        # Define the contract's ABI and address
        contract_abi = [
            {
                inputs: {},
                stateMutability: 'nonpayable',
                type: 'constructor',
                constant: undefined,
                payable: undefined
            },
            {
                inputs: {},
                name: 'data',
                outputs: {},
                stateMutability: 'view',
                type: 'function',
                constant: true,
                payable: undefined,
                signature: '0xf0ba8440'
            },
            {
                inputs: {},
                name: 'owner',
                outputs: {},
                stateMutability: 'view',
                type: 'function',
                constant: true,
                payable: undefined,
                signature: '0x8da5cb5b'
            },
            {
                inputs: {},
                name: 'kill',
                outputs: {},
                stateMutability: 'nonpayable',
                type: 'function',
                constant: undefined,
                payable: undefined,
                signature: '0x41c0e1b5'
            },
            {
                inputs: {},
                name: 'updateData',
                outputs: {},
                stateMutability: 'nonpayable',
                type: 'function',
                constant: undefined,
                payable: undefined,
                signature: '0x68446ead'
            },
            {
                inputs: {},
                name: 'readData',
                outputs: {},
                stateMutability: 'view',
                type: 'function',
                constant: true,
                payable: undefined,
                signature: '0xbef55ef3'
            }
        ]

        # Replace with the actual ABI
        contract_address = '0x7a87C70d8aE383847D31c17101b9Abe9b09D3d24'  # Replace with the actual contract address

        # Load the contract
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        data_to_push = json.dumps(model_params)

        print(type(data_to_push))

        # Example: Send data to a function named 'storeData' with a string parameter
        tx_hash = contract.functions.updateData(data_to_push).transact()

        # Print the transaction hash
        print("Transaction Hash:", tx_hash.hex())

        response = requests.put(request_url, json=request_body)
        print('Response received from updating central model params:', response)
        if response.status_code != 200:
            print('Error updating central model params. Error:', response.reason)
        else:
            print('Model params updated on central successfully')
        sys.stdout.flush()

    def can_do_training(self):
        return self.status == ClientStatus.IDLE

    def register(self):
        print('Registering in central node:', self.SERVER_URL)
        request_url = self.SERVER_URL + '/client'
        try:
            print('Doing request', request_url)
            response = requests.post(request_url, data={'client_url': self.client_url}, timeout=5)
            print('Response received from registration:', response)
            if response.status_code != 201:
                print('Cannot register client in the system at', request_url, 'error:', response.reason)
            else:
                print('Client registered successfully')
        except Timeout:
            print('Cannot register client in the central node, the central node is not responding')
        sys.stdout.flush()
