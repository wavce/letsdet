from abc import ABCMeta
from abc import abstractclassmethod
import tensorflow as tf


class Detector(metaclass=ABCMeta):
    def __init__(self, cfg, training=True):
        self.cfg = cfg
        self.training = training
    
    @abstractclassmethod
    def compute_losses(self, predictions, image_info):
        raise NotImplementedError()
    
    @abstractclassmethod
    def save_weights(self, name):
        raise NotImplementedError()

