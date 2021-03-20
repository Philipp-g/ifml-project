import copy
import pickle
from collections import namedtuple, deque
from typing import List, Tuple, Optional

import events as e
import torch
import numpy as np

import settings
from .callbacks import look_for_targets

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def augment_field(old_game_state : dict, new_game_state : dict) -> Tuple[dict, dict]:
    if old_game_state["step"] < 2:
        return [(old_game_state, new_game_state)]
    augmented_states = []
    simple_augmented_states = [(old_game_state,new_game_state)]
    field = old_game_state["field"]
    me = old_game_state["self"]
    x,y = me[3]
    own_bomb = ()
    for bomb in old_game_state["bombs"]:
        if bomb[2] == me[0]:
            own_bomb = bomb
    surrounding = [(x,y), (x,y+1), (x,y-1), (x+1, y), (x-1,y)]
    bomb_radius = []
    if own_bomb:
        bx,by = own_bomb[0]
        for i in range(-3,4):
            bomb_radius.append((bx+i, by))
            bomb_radius.append((bx, by + i))
    bomb_radius = [(x,y) for x,y in bomb_radius if x > 0 and x < 16 and y > 0 and y < 16]
    frozen_fields = surrounding + bomb_radius

    for other_old, other_new in zip(old_game_state["others"], new_game_state["others"]):
        old = copy.deepcopy(old_game_state)
        new = copy.deepcopy(new_game_state)
        if other_old[3] not in frozen_fields and other_new[3] not in frozen_fields:
            old["others"].remove(other_old)
            new["others"].remove(other_new)

    for bomb in old_game_state["bombs"]:
        old = copy.deepcopy(old_game_state)
        new = copy.deepcopy(new_game_state)
        if bomb[0] not in frozen_fields and bomb[1] > 1:
            old["bombs"].remove(bomb)
            new_bomb = (bomb[0], bomb[1]-1, bomb[2])
            new["bombs"].remove(new_bomb)
            # Set correct bomb boolean
            old_others, new_others = [],[]
            for other_o, other_n in zip(old["others"],new["others"]):
                if other_o == bomb[2]:
                    old_others.append((other_o[0],other_o[1],True, other_o[2]))
                    new_others.append((other_n[0], other_n[1], True, other_n[2]))
            old["others"] = old_others
            new["others"] = new_others
            simple_augmented_states.append((old, new))

    #mask = np.random.randint(0,2,size=(17,17)).astype(np.bool)
    frozen_fields = np.array(frozen_fields)
    #mask[frozen_fields[:,0], frozen_fields[:,1]] = False
    not_blocked = field != -1
    #combined_mask = np.logical_and(mask,not_blocked)
    #values = np.random.randint(0,2,size=(17,17))

    for i in range(32):
        num_crates = (old_game_state["field"] == 1).sum()
        mask = np.random.choice([1, 0], size=(17, 17), p=[num_crates / (15 * 15), 1. - num_crates / (15 * 15)])
        mask[frozen_fields[:, 0], frozen_fields[:, 1]] = False
        combined_mask = np.logical_and(mask, not_blocked)
        values = np.random.randint(0, 2, size=(17, 17))
        for old_simple, new_simple in simple_augmented_states:
            old_augmented = copy.deepcopy(old_simple)
            new_augmented = copy.deepcopy(new_simple)
            old_augmented["field"][combined_mask] = values[combined_mask]
            new_augmented["field"][combined_mask] = values[combined_mask]
            augmented_states.append((old_augmented, new_augmented))

    return augmented_states

def get_permutations(field):
    permutations = []
    for i in range(4):
        permutation = np.rot90(field, i)
        permutations.append(permutation)
    field = np.fliplr(field)
    for i in range(4):
        permutation = np.rot90(field, i)
        permutations.append(permutation)
    return permutations

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return
    if new_game_state is None:
        return
    if self_action is None:
        return

    augmented_states = augment_field(old_game_state, new_game_state)
    for old_state,_ in augmented_states:
        old_features, old_step = state_to_features(self, old_state)
        old_perm = get_permutations(old_features)
        self.states.extend(get_permutations(state_to_features(self, old_perm)))

def setup_training(self):
    self.others = {}
    self.states = []

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    round = last_game_state["round"]
    if round > 400:
        data = np.array(self.states)
        #data_unique = np.unique(data, axis=0)
        with open("states.npy", "wb") as f:
            np.save(f, data, allow_pickle=False)



def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    pass

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
        for i, (name, p, b, (x,y)) in enumerate(game_state["others"]):
            self.others[name] = i

    field_channel[game_state["self"][3]] = 3

    for name, p, b, (x, y) in game_state["others"]:
        field_channel[(x, y)] = self.others[name] + 4

    for (x, y), timer, owner in game_state["bombs"]:
        if owner == game_state["self"][0]:
            field_channel[(x, y)] = timer + 7
        else:
            field_channel[(x,y)] = timer + 11

    field_channel = field_channel[1:-1, 1:-1]
    field_channel_torch = torch.from_numpy(field_channel).float()
    normalized = (field_channel_torch + 1) / (14 + 1)
    step_t = torch.Tensor([game_state["step"]])
    return normalized.unsqueeze(0), step_t / 400
