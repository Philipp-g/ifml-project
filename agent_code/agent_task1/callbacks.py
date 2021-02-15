import math
import os
import pickle
import random

import numpy as np
import torch

from . import config
from .model import BombNet
from typing import Optional


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile(config.LOAD_PATH):
        self.logger.info("Setting up model from scratch.")
        if torch.cuda.is_available():
            self.model = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float().cuda()
        else:
            self.model = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float()
    else:
        self.logger.info("Loading model from saved state.")
        # TODO load with state_dict
        self.model = torch.load(config.LOAD_PATH)
        self.model.eval()
    self.explore_prob = config.MINIMAL_EXPLORE_PROB


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]
    current_step = game_state["step"]
    if self.train and random.random() < self.explore_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_index = np.random.choice(list(range(len(config.ACTIONS))), p=[.2, .2, .2, .2, .2])
    else:
        self.logger.debug("Querying model for action.")
        features = state_to_features(game_state)
        # add dimension for 1 batch
        features = features.unsqueeze(0)
        if torch.cuda.is_available():
            features = features.cuda()
        output = self.model(features, "online")

        action_index = output.argmax(1).item()

    # decay explore prob
    self.explore_prob = max(config.MINIMAL_EXPLORE_PROB,
                            config.INITIAL_EXPLORE_PROB * math.exp(
                                -current_step * current_round / config.EXPLORE_DECAY))
    self.logger.debug(f"ACT RETURN: {config.ACTIONS[action_index]}")
    return config.ACTIONS[action_index]


def state_to_features(game_state: dict) -> Optional[torch.Tensor]:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: torch.Tensor
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    field_channel = game_state["field"]
    explosion_channel = game_state["explosion_map"]
    coins_channel = np.zeros_like(field_channel)
    bombs_channel = np.zeros_like(field_channel)
    others_channel = np.zeros_like(field_channel)
    self_channel = np.zeros_like(field_channel)

    coins = np.array(game_state["coins"])

    for (x, y), time in game_state["bombs"]:
        bombs_channel[x, y] = time
    for name, score, bomb_state, (x, y) in game_state["others"]:
        others_channel[x, y] = bomb_state

    # onehot encode coins
    coins_channel[coins[:, 0], coins[:, 1]] = 1

    #self_channel[game_state["self"][3]] = int(game_state["self"][2])
    self_channel[game_state["self"][3]] = 1

    #channels.extend([field_channel, explosion_channel, coins_channel, bombs_channel, others_channel, self_channel])
    channels.extend([field_channel, coins_channel, self_channel])
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # stacked_channels = stacked_channels[np.newaxis, :, :, :]
    stacked_torch = torch.from_numpy(stacked_channels).float()
    return stacked_torch
    # and return them as a vector
    return stacked_channels  # .reshape(-1)
