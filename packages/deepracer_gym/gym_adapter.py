import zmq
import numpy as np
from typing import TypeAlias
from collections.abc import Callable

from deepracer_gym.zmq_client import DeepracerClientZMQ
from deepracer_gym.utils import (
    terminated_check, truncated_check
)


PORT: int=8888
HOST: str='127.0.0.1'
TIMEOUT_LONG: int=500_000   # ~8.3 m
TIMEOUT_SHORT: int=100_000  # ~1.7 m
DUMMY_ACTION_DISCRETE: Callable[[], int]=(
    lambda: 0
)
DUMMY_ACTION_CONTINUOUS: Callable[[], np.ndarray[float]]=(
    lambda: np.random.uniform(-1, 1, 2)
)
ActionType: TypeAlias=(int | np.ndarray | list[float])

class DeepracerGymAdapter:
    def __init__(
            self,
            action_space_type: str,
            host: str=HOST,
            port: int=PORT):
        if action_space_type == 'discrete':
            self.dummy_action = DUMMY_ACTION_DISCRETE
        elif action_space_type == 'continuous':
            self.dummy_action = DUMMY_ACTION_CONTINUOUS
        else:
            raise ValueError(
                f'Action space can only be discrete or continuous. Got {action_space_type} instead.'
            )
        self.zmq_client = DeepracerClientZMQ(host=host, port=port)
        self.zmq_client.ready()
        self.response = None
        self.done = False

    def _send_action(self, action: ActionType):
        action: dict[str, ActionType] = {'action': action}
        self.response = self.zmq_client.send_message(action)
        self.done = self.response['_game_over']
        return self.response
    
    def env_reset(self):
        if self.response is None:
            # First communication to zmq server
            self.response = self.zmq_client.recieve_response()
            # Smaller timeout after first connection
            self.zmq_client.socket.set(zmq.SNDTIMEO, TIMEOUT_SHORT)
            self.zmq_client.socket.set(zmq.RCVTIMEO, TIMEOUT_SHORT)
        elif self.done:
            pass
        else:
            while not self.done:
                self.response = self._send_action(self.dummy_action())
        
        if not isinstance(self.response['info'], dict):
            self.response['info'] = dict()
        
        # If prev_episode done and reset called, fast forward one step for new episode
        # dummy action ignored due to reset()
        step = (
            self.response['info']['reward_params']['steps']
        )
        while step != 1:
            self.response = self._send_action(self.dummy_action())
            step = (
                self.response['info']['reward_params']['steps']
            )

        observation, _, _, _, info = self._parse_response(self.response)
        return observation, info
    
    def send_action(self, action: ActionType):
        if self.done:
            return self._parse_response(self.response)
        response = self._send_action(action)
        return self._parse_response(response)
    
    @staticmethod
    def _parse_response(response: dict):
        info = response['info']
        if not isinstance(info, dict):
            info = dict()
        info['goal'] = response['_goal']

        game_over = response['_game_over']
        terminated = terminated_check(info['episode_status'], game_over)
        truncated = truncated_check(info['episode_status'], game_over)
        
        reward = response['_reward']
        observation = response['_next_state']
        # channel first convention
        observation = {
            sensor: (
                measurement.transpose(-1, 0, 1) if 'CAMERA' in sensor
                else measurement
            ) for sensor, measurement in observation.items()
        }

        return observation, reward, terminated, truncated, info
