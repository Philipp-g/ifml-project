import copy
import random
from collections import namedtuple, deque
import datetime
from typing import List, Tuple, Any, Union
from torch import Tensor

import events as e
from .callbacks import state_to_features
import torch
from collections import defaultdict
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.loss_fn = torch.nn.SmoothL1Loss()
    self.loss = []
    self.event_logger = defaultdict(int)
    self.global_logger = defaultdict(int)
    self.positions = []
    self.reward = []
    self.in_danger = False
    self.enemy_danger = defaultdict(bool)
    self.bomb_history = deque(maxlen=5)
    self.best_loss = np.inf
    self.total_steps = 0


def reset(self):
    self.loss = []
    self.event_logger = defaultdict(int)
    self.model_actions = 0
    self.positions = []
    self.reward = []
    self.in_danger = False
    self.enemy_danger = defaultdict(bool)
    self.bomb_history = deque(maxlen=5)


def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: List[str]):
    pass

def in_danger(position, bombs):
    x, y = position
    for (xb, yb), *_ in bombs:
        if xb == x and abs(yb - y) < 4:
            return True
        elif yb == y and abs(xb - x) < 4:
            return True
    return False


def reachable_tiles(game_state):
    position = game_state["self"][3]
    field = game_state["field"]
    bombs = game_state["bombs"]
    others = [agent[3] for agent in game_state["others"]]
    stack = [(position, 0)]
    global_stack = []
    while stack:
        (x, y), depth = stack.pop(0)
        valid_tiles = []
        surrounding_tiles = [(x, y), (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        for possible in surrounding_tiles:
            if field[possible] == 0 and possible not in others:
                flag = True
                for (xb, yb), *_ in bombs:
                    if possible[0] == xb and possible[1] == yb:
                        flag = False
                if flag and depth < 4:
                    valid_tiles.append((possible, depth + 1))
        global_stack.extend([valid[0] for valid in valid_tiles])
        stack.extend(valid_tiles)
    return set(global_stack)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        return
    if new_game_state is None:
        return
    position = new_game_state["self"][3]
    x, y = position
    if position in self.positions[:-5]:
        events.append(e.LOOPED)
    if self_action == "BOMB" and e.INVALID_ACTION not in events:
        if position in self.bomb_history:
            events.append(e.BOMB_LOOP)
        self.bomb_history.append(position)

    self.positions.append(position)

    step = old_game_state["step"]
    round = old_game_state["round"]

    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (old_game_state["field"][x, y] == 1)]
    others = [(x, y) for (name, p, b, (x, y)) in old_game_state["others"]]
    target = look_for_targets(old_game_state["field"] == 0, old_game_state["self"][3],
                              old_game_state["coins"] + others)
    bomb_owners = [owner for (xb, yb), t, owner in old_game_state["bombs"]]

    if self_action == "BOMB":
        self.in_danger = True
        if new_game_state["self"][0] not in bomb_owners:
            for (xb, yb), t, owner in new_game_state["bombs"]:
                if owner == new_game_state["self"][0]:
                    for xo, yo in others:
                        dist = np.sqrt((xo - xb) ** 2 + (yo - yb) ** 2)
                        bomb_other_danger = False
                        if dist <= 5:
                            events.append(e.BOMB_CLOSE_TO_TARGET)
                        if xb == xo and abs(yb - yo) < 4:
                            bomb_other_danger = True
                        elif yb == yo and abs(xb - xo) < 4:
                            bomb_other_danger = True
                        if bomb_other_danger:
                            events.append(e.BOMB_OTHER_DANGER)

    for (xb, yb), *_ in new_game_state["bombs"]:
        if xb == x and abs(yb - y) < 4:
            if self.in_danger:
                events.append(e.IN_DANGER)
            else:
                self.in_danger = True
                events.append(e.IN_DANGER)
                events.append(e.MOVED_INTO_DANGER)
        elif yb == y and abs(xb - x) < 4:
            if self.in_danger:
                events.append(e.IN_DANGER)
            else:
                self.in_danger = True
                events.append(e.IN_DANGER)
                events.append(e.MOVED_INTO_DANGER)
        else:
            if self.in_danger:
                events.append(e.MOVED_OUT_OF_DANGER)
            self.in_danger = False

    if target == new_game_state["self"][3]:
        events.append(e.CLOSER_TO_TARGET)
    else:
        events.append(e.FARTHER_FROM_TARGET)
    reachable = reachable_tiles(old_game_state)
    own_bomb = [bomb for bomb in new_game_state["bombs"] if bomb[2] == new_game_state["self"][0]]
    trapped = all([in_danger(tile, own_bomb) for tile in reachable])
    if trapped:
        events.append(e.TRAPPED)
    for event in events:
        self.event_logger[event] += 1
    rewards = reward_from_events(self, events)
    self.reward.append(rewards)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    steps = last_game_state["step"]
    round = last_game_state["round"]

    for event in events:
        self.event_logger[event] += 1

    rewards = reward_from_events(self, events)
    self.reward.append(rewards)

    for event in dir(e):
        if not event.startswith("__"):
            self.global_logger[event] += self.event_logger[event]
            if last_game_state["round"] == 100:
                self.global_logger[event] /= 100

    if last_game_state["round"] == 100:
        print(self.global_logger)

    # Reset round specific objects
    reset(self)

def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -1,
        e.INVALID_ACTION: -2,
        e.KILLED_SELF: -8,
        e.CRATE_DESTROYED: 1,
        #e.BOMB_OTHER_DANGER: 4,
        e.BOMB_DROPPED: 0.2,
        #e.BOMB_CLOSE_TO_TARGET: 1,
        e.TRAPPED: -5,
        e.CLOSER_TO_TARGET: 0.05,
        e.COIN_COLLECTED: 4,
        e.KILLED_OPPONENT: 20,
        e.GOT_KILLED: -4
    }
    reward_sum = -0.05

    if e.IN_DANGER in events:
        game_rewards[e.INVALID_ACTION] = -5
    if e.KILLED_SELF in events or e.INVALID_ACTION in events:
        for event in events:
            if event in game_rewards:
                if game_rewards[event] < 0:
                    reward_sum += game_rewards[event]
        return reward_sum
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
