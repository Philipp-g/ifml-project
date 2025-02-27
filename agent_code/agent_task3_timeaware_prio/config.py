FEATURE_SHAPE = (1, 15, 15)
BATCH_SIZE = 64
INPUT_DIMS = (BATCH_SIZE, *FEATURE_SHAPE)
MODEL_NAME = "AGENT_TASK3_SINGLE_CHANNEL_TIMEAWARE_PRIO"
SAVE_PATH = f"./{MODEL_NAME}-"
LOAD_PATH = ""
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
INITIAL_EXPLORE_PROB = 0.3
MINIMAL_EXPLORE_PROB = 0.1
EXPLORE_DECAY = 0.99999555
TRANSITION_MEM_SIZE = 500_000
GAMMA = 0.95
MINIMAL_TRANSITION_LEN = 25_000
UPDATE_MINIBATCH = 4
UPDATE_TARGET_STEPS = 25_000
SAVE_STEPS = 50_000
BOMB_HISTORY_SIZE = 5
