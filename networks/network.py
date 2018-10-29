from abc import ABC, abstractmethod
from logger import get_logger

class Network:
    def __init__(self):
        self.logger = get_logger()

    @abstractmethod
    def create_network(self, features, labels, seq_len, num_classes, is_training):
        pass

    @abstractmethod
    def validate(self, mfccs, labels, seq_len, transcripts):
        pass

    @abstractmethod
    def evaluate(self, mfccs, labels, seq_len, transcripts):
        pass

    @abstractmethod
    def decode(self, mfccs, seq_len):
        pass

    @abstractmethod
    def train(self, mfccs, labels, seq_len, transcripts):
        pass