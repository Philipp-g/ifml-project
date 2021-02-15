import pickle
import random
from collections import namedtuple, deque
import datetime
from typing import List, Tuple

import events as e
from .callbacks import state_to_features
from . import config
import torch
from collections import defaultdict
import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=config.TRANSITION_MEM_SIZE)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0025)
    self.loss_fn = torch.nn.SmoothL1Loss()
    self.loss = defaultdict(list)


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

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state is None:
        return
    if new_game_state is None:
        return

    step = old_game_state["step"]
    round = old_game_state["round"]
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))

    if len(self.transitions) < config.MINIMAL_TRANSITION_LEN:
        return

    # Update target model every x steps
    # if step % config.UPDATE_TARGET_STEPS == 0:
    #     update_target(self)

    state, next_state, action, reward = sample_from_transitions(self, config.BATCH_SIZE)
    td_est = td_estimate(self, state, action)
    td_targ = td_target(self, reward, next_state)
    loss = update_online(self, td_est, td_targ)
    self.loss[round].append(loss)


def td_estimate(self, state, action):
    current_Q = self.model(state, model="online")
    current_Q_indexed = current_Q[torch.arange(0, config.BATCH_SIZE, dtype=torch.long), action]  # .squeeze(1)]
    return current_Q_indexed


def update_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()


def update_target(self):
    online = self.model.online.state_dict()
    self.model.target.load_state_dict(online)


@torch.no_grad()
def td_target(self, reward, next_state):
    next_state_Q = self.model(next_state, model="online")
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.model(next_state, model="target")
    next_Q_indexed = next_Q[torch.arange(0, config.BATCH_SIZE, dtype=torch.long), best_action]
    return (reward + config.GAMMA * next_Q_indexed).float()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # update target model
    update_target(self)
    steps = last_game_state["step"]
    round = last_game_state["round"]
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    print(f"Round: {last_game_state['round']}, Steps: {last_game_state['step']}")
    print(f"Mean Loss: {np.mean(self.loss[round])}")
    print(f"Points: {last_game_state['self'][1]}")

    #self.transitions.append(
    #    Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    save_path = config.SAVE_PATH + str(datetime.datetime.now())[:-7]
    torch.save(self.model, save_path)


def sample_from_transitions(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = random.sample(self.transitions, batch_size)
    states = []
    next_states = []
    actions = []
    rewards = []
    for transition in batch:
        states.append(transition.state)
        next_states.append(transition.next_state)
        actions.append(config.ACTIONS.index(transition.action))
        rewards.append(transition.reward)
    state = torch.stack(tuple(states))
    next_state = torch.stack(tuple(next_states))
    action = torch.Tensor(actions).long()  # .unsqueeze(1)
    reward = torch.Tensor(rewards)  # .unsqueeze(1)

    if torch.cuda.is_available():
        return state.cuda(), next_state.cuda(), action.cuda(), reward.cuda()
    else:
        return state, next_state, action, reward


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.SURVIVED_ROUND: 2000,
        e.WAITED: -10,
        e.INVALID_ACTION: -100,
        # e.KILLED_OPPONENT: 5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    # punish steps without events
    reward_sum = -1
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
