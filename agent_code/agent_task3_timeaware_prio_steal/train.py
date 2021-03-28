#Inspired by https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

import copy
import random
from collections import namedtuple, deque
import datetime
from typing import List, Tuple, Any, Union
from torch import Tensor

import events as e
from .callbacks import state_to_features
from . import config
import torch
from collections import defaultdict
import numpy as np

from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=config.TRANSITION_MEM_SIZE)
    self.enemy_transitions = deque(maxlen=2 * config.TRANSITION_MEM_SIZE)
    self.optimizer = torch.optim.Adam(self.model_online.parameters(), lr=0.0002)
    self.loss_fn = torch.nn.SmoothL1Loss()
    self.loss = []
    self.event_logger = defaultdict(int)
    self.positions = []
    self.reward = []
    self.in_danger = False
    self.enemy_danger = defaultdict(bool)
    self.bomb_history = deque(maxlen=config.BOMB_HISTORY_SIZE)
    self.writer = SummaryWriter("../../runs/" + config.MODEL_NAME + '_' + datetime.datetime.now().strftime('%D-%T'))
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
    self.bomb_history = deque(maxlen=config.BOMB_HISTORY_SIZE)


def get_permutations(game_state):
    permutations = []
    for i in range(4):
        permutation = torch.rot90(game_state, i, (1, 2))
        permutations.append(permutation)
    game_state = torch.fliplr(game_state)
    for i in range(4):
        permutation = torch.rot90(game_state, i, (1, 2))
        permutations.append(permutation)
    return permutations


def get_permutations_action(self_action):
    if self_action in ["WAIT", "BOMB"]:
        return 8 * [self_action]
    action_index = config.ACTIONS.index(self_action)
    permutations = []
    for i in range(4):
        permutations.append(config.ACTIONS[(action_index + i) % 4])
    # lr flip
    if action_index % 2 != 0:
        action_index = action_index + 2
    for i in range(4):
        permutations.append(config.ACTIONS[(action_index + i) % 4])
    return permutations


def augment_field(old_game_state: dict, new_game_state: dict) -> Tuple[dict, dict]:
    if old_game_state["step"] < 2:
        return [(old_game_state, new_game_state)]
    augmented_states = []
    simple_augmented_states = [(old_game_state, new_game_state)]
    frozen_fields = []
    field = old_game_state["field"]
    me = old_game_state["self"]
    x, y = me[3]
    own_bomb = ()
    for bomb in old_game_state["bombs"]:
        if bomb[2] == me[0]:
            own_bomb = bomb
    surrounding = [(x, y), (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]
    bomb_radius = []
    if own_bomb:
        bx, by = own_bomb[0]
        for i in range(-3, 4):
            bomb_radius.append((bx + i, by))
            bomb_radius.append((bx, by + i))
    bomb_radius = [(x, y) for x, y in bomb_radius if 0 < x < 16 and 0 < y < 16]
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
            new_bomb = (bomb[0], bomb[1] - 1, bomb[2])
            new["bombs"].remove(new_bomb)
            # Set correct bomb boolean
            old_others, new_others = [], []
            for other_o, other_n in zip(old["others"], new["others"]):
                if other_o == bomb[2]:
                    old_others.append((other_o[0], other_o[1], True, other_o[2]))
                    new_others.append((other_n[0], other_n[1], True, other_n[2]))
            old["others"] = old_others
            new["others"] = new_others
            simple_augmented_states.append((old, new))

    frozen_fields = np.array(frozen_fields)
    not_blocked = field != -1

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


def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: List[str]):
    if old_enemy_game_state is None:
        return
    if enemy_game_state is None:
        return
    if enemy_action is None:
        return
    position = enemy_game_state["self"][3]
    others = [(x, y) for (name, p, b, (x, y)) in enemy_game_state["others"]]
    bomb_owners = [owner for (xb, yb), t, owner in old_enemy_game_state["bombs"]]

    if enemy_action == "BOMB":
        self.in_danger = True
        if enemy_game_state["self"][0] not in bomb_owners:
            for (xb, yb), t, owner in enemy_game_state["bombs"]:
                if owner == enemy_game_state["self"][0]:
                    for xo, yo in others:
                        dist = np.sqrt((xo - xb) ** 2 + (yo - yb) ** 2)
                        bomb_other_danger = False
                        if dist <= 5:
                            enemy_events.append(e.BOMB_CLOSE_TO_TARGET)
                        if xb == xo and abs(yb - yo) < 4:
                            bomb_other_danger = True
                        elif yb == yo and abs(xb - xo) < 4:
                            bomb_other_danger = True
                        if bomb_other_danger:
                            enemy_events.append(e.BOMB_OTHER_DANGER)

    step = old_enemy_game_state["step"]
    round = old_enemy_game_state["round"]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (old_enemy_game_state["field"][x, y] == 1)]
    others = [(x, y) for (name, p, b, (x, y)) in old_enemy_game_state["others"]]
    target = look_for_targets(old_enemy_game_state["field"] == 0, old_enemy_game_state["self"][3],
                              old_enemy_game_state["coins"] + others)
    if target == enemy_game_state["self"][3]:
        enemy_events.append(e.CLOSER_TO_TARGET)
    else:
        enemy_events.append(e.FARTHER_FROM_TARGET)
    rewards = reward_from_events(self, enemy_events)
    old_features, old_step = state_to_features(self, old_enemy_game_state)
    new_features, new_step = state_to_features(self, enemy_game_state)
    old_perm = get_permutations(old_features)
    new_perm = get_permutations(new_features)
    action_perm = get_permutations_action(enemy_action)
    for old, action, new in zip(old_perm, action_perm, new_perm):
        self.enemy_transitions.append(Transition((old, old_step), action, (new, new_step), rewards, 0))


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
    old_features, old_step = state_to_features(self, old_game_state)
    new_features, new_step = state_to_features(self, new_game_state)
    old_perm = get_permutations(old_features)
    new_perm = get_permutations(new_features)
    action_perm = get_permutations_action(self_action)
    for old, action, new in zip(old_perm, action_perm, new_perm):
        self.transitions.append(Transition((old, old_step), action, (new, new_step), rewards, 0))
    self.total_steps += 1
    if len(self.transitions) < config.MINIMAL_TRANSITION_LEN:
        return

    if self.total_steps % config.UPDATE_MINIBATCH == 0:
        state, next_state, action, reward, done = sample_from_transitions(self, config.BATCH_SIZE)
        td_est = td_estimate(self, state, action)
        td_targ = td_target(self, reward, next_state, done)
        loss = update_online(self, td_est, td_targ)
        self.loss.append(loss)

    if self.total_steps % config.UPDATE_TARGET_STEPS == 0:
        # update target model
        update_target(self)

    # Store the model
    if self.total_steps % config.SAVE_STEPS == 0:
        save_path = config.SAVE_PATH + str(datetime.datetime.now())[:-7] + ".mdl"
        torch.save({
            'round': round,
            'model_state_dict': self.model_online.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)


def td_estimate(self, state, action):
    current_Q = self.model_online(state)
    current_Q_indexed = current_Q[torch.arange(0, action.shape[0], dtype=torch.long), action]  # .squeeze(1)]
    return current_Q_indexed


def update_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()


def update_target(self):
    online = self.model_online.state_dict()
    self.model_target.load_state_dict(online)


@torch.no_grad()
def td_target(self, reward, next_state, done):
    next_state_Q = self.model_online(next_state)
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.model_target(next_state)
    next_Q_indexed = next_Q[torch.arange(0, done.shape[0], dtype=torch.long), best_action]
    return (reward + (1 - done.float()) * config.GAMMA * next_Q_indexed).float()


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
    old_features, old_step = state_to_features(self, last_game_state)
    new_features, new_step = torch.zeros_like(old_features), old_step + 1
    old_perm = get_permutations(old_features)
    action_perm = get_permutations_action(last_action)
    for old, action in zip(old_perm, action_perm):
        self.transitions.append(Transition((old, old_step), action, (new_features, new_step), rewards, 1))

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.writer.add_scalar("Loss", np.mean(self.loss), round)
    self.writer.add_scalar("Points", last_game_state['self'][1], round)
    for event in dir(e):
        if not event.startswith("__"):
            self.writer.add_scalar(event, self.event_logger[event], round)

    self.writer.add_scalar("Total Actions", steps, round)
    self.writer.add_scalar("Model Actions", self.model_actions, round)
    self.writer.add_scalar("Avg. Reward", np.mean(self.reward), round)
    self.writer.add_scalar("Total Reward", np.sum(self.reward), round)
    self.writer.flush()
    # Reset round specific objects
    reset(self)


def sample_from_transitions(self, batch_size: int) -> Union[
    tuple[tuple[Any, Tensor], tuple[Any, list], Any, Any, Any], tuple[
        tuple[Tensor, Tensor], tuple[Tensor, list], Any, Tensor, Tensor]]:
    # Sample from a different transition buffer with prob 0.1
    if random.random() < 0.1:
        batch = random.sample(self.enemy_transitions, batch_size * 4)
    else:
        batch = random.sample(self.transitions, batch_size * 4)

    states = []
    states_steps = []
    next_states = []
    next_states_steps = []
    actions = []
    rewards = []
    dones = []
    for transition in batch:
        states.append(transition.state[0])
        states_steps.append(transition.state[1])
        next_states.append(transition.next_state[0])
        next_states_steps.append(transition.next_state[1])
        actions.append(config.ACTIONS.index(transition.action))
        rewards.append(transition.reward)
        dones.append(transition.done)
    state = torch.stack(tuple(states)).cuda()
    states_steps = torch.Tensor(states_steps).unsqueeze(1).cuda()
    next_state = torch.stack(tuple(next_states)).cuda()
    next_states_steps = torch.Tensor(next_states_steps).unsqueeze(1).cuda()
    action = torch.Tensor(actions).long().cuda()
    reward = torch.Tensor(rewards).cuda()
    done = torch.Tensor(dones).cuda()
    with torch.no_grad():
        td_est = td_estimate(self, (state, states_steps), action)
        td_targ = td_target(self, reward, (next_state, next_states_steps), done)
        loss_fn = torch.nn.SmoothL1Loss(reduction="none")
        loss = loss_fn(td_est, td_targ)
        _, top_batch = torch.topk(loss, k=batch_size)
    return (state[top_batch], states_steps[top_batch]), (
        next_state[top_batch], next_states_steps[top_batch]), action[top_batch], reward[top_batch], done[top_batch]


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
