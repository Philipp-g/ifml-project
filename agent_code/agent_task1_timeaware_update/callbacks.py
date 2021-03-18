import copy
import os
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
    if self.train and not os.path.isfile(config.LOAD_PATH):
        self.logger.info("Setting up model from scratch.")
        if torch.cuda.is_available():
            self.model_online = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float().cuda()
        else:
            self.model_online = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float()
    else:
        self.logger.info("Loading model from saved state.")
        # TODO CUDA DISTINCTION
        self.model_online = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float().cuda()
        checkpoint = torch.load(config.LOAD_PATH)
        self.model_online.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.model.eval()
    self.explore_prob = config.INITIAL_EXPLORE_PROB
    self.model_actions = 0
    self.model_target = copy.deepcopy(self.model_online)

def get_allowed(game_state: dict):
    x, y = game_state["self"][3]
    field = game_state["field"]
    surrounding_tiles = [(x, y), (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]

    valid_tiles = []
    valid_actions = []
    for possible in surrounding_tiles:
        if field[possible] == 0:
            valid_tiles.append(possible)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    if game_state["self"][1] == 0:
        valid_actions.append('BOMB')
    return valid_actions


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]
    current_step = game_state["step"]


    if self.train and random.random() < self.explore_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        possible_actions = [config.ACTIONS.index(action) for action in get_allowed(game_state)]
        move_actions = [possible_action for possible_action in possible_actions if possible_action < 4]
        probs_move = (1 - (len(possible_actions) - len(move_actions)) / 10) / len(move_actions)
        prob_no_move = (len(possible_actions) - len(move_actions)) / 10 / (len(possible_actions) - len(move_actions))
        probs = []
        for action in possible_actions:
            if action > 3:
                probs.append(prob_no_move)
            else:
                probs.append(probs_move)
        action_index = np.random.choice([config.ACTIONS.index(action) for action in get_allowed(game_state)], p=probs)
    else:
        self.logger.debug("Querying model for action.")
        features, step_t = state_to_features(game_state)
        # add dimension for 1 batch
        features = features.unsqueeze(0)
        step_t = step_t.unsqueeze(0)
        if torch.cuda.is_available():
            features = features.cuda()
            step_t = step_t.cuda()
        output = self.model_online((features, step_t))
        action_index = output.argmax(1).item()
        self.model_actions += 1

    self.explore_prob *= config.EXPLORE_DECAY
    self.explore_prob = max(config.MINIMAL_EXPLORE_PROB, self.explore_prob)
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

    field_channel = game_state["field"]

    coins = np.array(game_state["coins"])

    try:
        field_channel[coins[:, 0], coins[:, 1]] = 2
    except:
        pass

    field_channel[game_state["self"][3]] = 3
    for (x, y), timer in game_state["bombs"]:
        field_channel[(x, y)] = timer + 4

    field_channel = field_channel[1:-1, 1:-1]
    field_channel_torch = torch.from_numpy(field_channel).float()
    #normalized = (field_channel_torch - field_channel_torch.min())/(field_channel_torch.max() - field_channel_torch.min())
    normalized = (field_channel_torch + 1) / (7 + 1)
    step_t = torch.Tensor([game_state["step"]])
    return normalized.unsqueeze(0), step_t / 400
