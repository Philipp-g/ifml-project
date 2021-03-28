
import numpy as np
import torch
import inspect
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

    self.logger.info("Loading model from saved state.")
    self.model_online = BombNet((1, 1, 15, 15), 6).float()
    checkpoint = torch.load("model.mdl", map_location=torch.device('cpu'))
    self.model_online.load_state_dict(checkpoint['model_state_dict'])
    self.model_actions = 0
    self.others = {}
    self.my_bombs = {}
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def act(self, game_state: dict) -> str:
    call_stack = inspect.stack()
    bombs = inspect.getargvalues(call_stack[5][0]).locals["self"].bombs
    bomb_list = []
    for bomb in bombs:
        bomb_list.append(((bomb.x, bomb.y), bomb.timer, bomb.owner.name))
    game_state["bombs"] = bomb_list

    current_round = game_state["round"]
    current_step = game_state["step"]

    self.logger.debug("Querying model for action.")
    features, step_t = state_to_features(self, game_state)
    # add dimension for 1 batch
    features = features.unsqueeze(0)
    step_t = step_t.unsqueeze(0)
    output = self.model_online((features, step_t))
    action_index = output.argmax(1).item()
    self.model_actions += 1
    self.logger.debug(f"ACT RETURN: {self.ACTIONS[action_index]}")
    return self.ACTIONS[action_index]


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
