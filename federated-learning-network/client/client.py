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

from web3 import Web3, Account

compiled_contract_path = '../../smart-contracts/build/contracts/SmartContract.json'


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
        with open(compiled_contract_path) as file:
            contract_json = json.load(file)  # load contract info as JSON
            contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions

        # Generate a new Ethereum account
        new_account = Account.create()

        # Retrieve the address and private key of the new account
        address = new_account.address
        private_key = new_account.key.hex()

        # Print the address and private key
        print("Address:", address)
        print("Private Key:", private_key)

        contract_address = '0xaC1727eD7F6e36d1eC38688D13aEBD464F68d588'  # Replace with the actual contract address
        #account_address = '0x2331D809c74C4A26078bfD35c14D90483f086246'
        #sender_address = w3.eth.defaultAccount = w3.eth.accounts[0]
        #private_key = '156be55f43e0516b326bfc54de56c0e9b57af20925a5d3d2857279adc96b140c'

        contract = w3.eth.contract(address=contract_address, abi=contract_abi)


        # Example: Send data to a function named 'storeData' with a string parameter
        nonce = w3.eth.get_transaction_count(address)
        transaction = contract.functions.updateData('Flipastulipas').build_transaction({
            'from': address,
            'gas': 88968,#21628,  # Adjust the gas limit as per your requirement
            'nonce': nonce,
        })

        gas_estimate = w3.eth.estimate_gas(transaction)
        print("Gas estimate:", gas_estimate)

        # Sign the transaction
        signed_transaction = w3.eth.account.sign_transaction(transaction, private_key=private_key)

        # Send the transaction
        tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

        # Wait for the transaction to be mined
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(tx_receipt)

        # Retrieve the updated data from the contract
        result = contract.functions.readData().call()

        # Print the result
        print(result)

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
