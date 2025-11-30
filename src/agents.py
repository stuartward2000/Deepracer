import abc
import torch
import torch.nn as nn

from src.transforms import EncodeObservation


class Agent(nn.Module, abc.ABC):
    '''Boilerplate class for providing interface'''
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    @abc.abstractmethod
    def get_action(self, x):
        raise NotImplementedError


class RandomAgent(Agent):
    '''
    A random agent for demonstrating usage of the environment
    '''
    def __init__(self, environment, name='random'):
        super().__init__(name=name)
        self.action_space = environment.action_space        

    def get_action(self, x):
        return self.action_space.sample()


class MyFancyAgent(Agent):
    '''
    Your own deepracer agent.
    '''
    def __init__(self, name='my_fancy_agent'):
        super().__init__(name=name)
        
        # in case you want to modify/ change/ transform
        # the observation (of course we have to un-flatten it).
        self.encoder = EncodeObservation()

        # anything else you may need...
        raise NotImplementedError
    
    def get_action(self, x):

        # un-flatten observation, and other transforms if required.
        z = self.encoder(x)

        # implement your fancy policy
        raise NotImplementedError