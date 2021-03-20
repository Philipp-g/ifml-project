import copy
import os
import random

import numpy as np
import torch

from . import config
from .model import BombNet, AutoEncoder
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
            self.encoder = AutoEncoder().cuda()
        else:
            self.model_online = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float()
            self.encoder = AutoEncoder()
            checkpoint_encoder = torch.load("../../autoencoder/autoencoder.pth")
            # Throw away decoder weights
            self.encoder.load_state_dict(checkpoint_encoder, strict=False)
    else:
        self.logger.info("Loading model from saved state.")
        # TODO CUDA DISTINCTION
        self.model_online = BombNet(config.INPUT_DIMS, len(config.ACTIONS)).float().cuda()
        checkpoint = torch.load(config.LOAD_PATH)
        self.model_online.load_state_dict(checkpoint['model_state_dict'])
        self.encoder = AutoEncoder().cuda()
        checkpoint_encoder = torch.load("../../autoencoder/autoencoder.pth")
        # Throw away decoder weights
        self.encoder.load_state_dict(checkpoint_encoder, strict=False)
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.model.eval()
    self.explore_prob = config.INITIAL_EXPLORE_PROB
    self.model_actions = 0
    self.model_target = copy.deepcopy(self.model_online)

def get_allowed(game_state: dict):
    x, y = game_state["self"][3]
    field = game_state["field"]
    surrounding_tiles = [(x, y), (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]

    bombs = game_state["bombs"]

    valid_tiles = []
    valid_actions = []
    for possible in surrounding_tiles:
        if field[possible] == 0:
            flag = True
            for (xb,yb), *_ in bombs:
                if xb == possible[0] and abs(yb - possible[1]) < 4:
                    flag = False
                elif yb == possible[1] and abs(xb - possible[0]) < 4:
                    flag = False
            if flag:
                valid_tiles.append(possible)
    # If there is no certain action which doesnt kill us pick any action
    if not valid_tiles:
        for possible in surrounding_tiles:
            if field[possible] == 0:
                if possible not in [(x,y) for (x,y),*_ in bombs]:
                    valid_tiles.append(possible)

    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    if game_state["self"][2]:
        valid_actions.append('BOMB')
    if not valid_actions:
        return ["WAIT"]
    return valid_actions


def act(self, game_state: dict) -> str:
    current_round = game_state["round"]
    current_step = game_state["step"]


    if self.train and random.random() < self.explore_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_index = np.random.choice([config.ACTIONS.index(action) for action in get_allowed(game_state)])
    else:
        self.logger.debug("Querying model for action.")
        features, step_t = state_to_features(self, game_state)
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


def state_to_features(self, game_state: dict) -> Optional[torch.Tensor]:
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

    field_channel = game_state["field"].copy()

    coins = np.array(game_state["coins"])

    try:
        field_channel[coins[:, 0], coins[:, 1]] = 2
    except:
        pass

    if game_state["step"] == 1:
        self.others = {}
        for i, (name, p, b, (x, y)) in enumerate(game_state["others"]):
            self.others[name] = i

    field_channel[game_state["self"][3]] = 3

    for name, p, b, (x, y) in game_state["others"]:
        field_channel[(x, y)] = self.others[name] + 4

    for (x, y), timer, owner in game_state["bombs"]:
        if owner == game_state["self"][0]:
            field_channel[(x, y)] = timer + 7
        else:
            field_channel[(x, y)] = timer + 11

    field_channel = field_channel[1:-1, 1:-1]
    field_channel_torch = torch.from_numpy(field_channel).float()
    normalized = (field_channel_torch + 1) / (14 + 1)
    step_t = torch.Tensor([game_state["step"]])
    return normalized.unsqueeze(0), step_t / 400
