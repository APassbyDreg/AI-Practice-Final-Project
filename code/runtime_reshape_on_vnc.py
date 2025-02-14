import shutil
import torch
import json
import random
import numpy as np
import os
import sys
import logging
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm


try:
    from code.models import *
    from code.tools import *
    from code.model_sl_tools import save_ckpt
    from source import malmoutils
except:
    from tools import *
    from models import *
    from model_sl_tools import save_ckpt
    import malmoutils

try:
    from malmo import MalmoPython
except:
    import MalmoPython


##################################### set logger
if not os.path.exists("./logs"):
    os.makedirs("./logs")
timestamp = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("./logs/{}.log".format(timestamp))
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
################################################


################################## prepare malmo
agent_host = MalmoPython.AgentHost()
mission_file = "./mazes"
# schema_dir = None
# try:
#     schema_dir = "mazes"
# except KeyError:
#     logger.info("MALMO_XSD_PATH not set? Check environment.")
#     exit(1)
# mission_file = os.path.abspath(schema_dir)
# if not os.path.exists(mission_file):
#     logger.info("Could not find Maze.xml under MALMO_XSD_PATH")
#     exit(1)
# add some args
agent_host.addOptionalStringArgument('mission_file',
                                     'Path/to/file from which to load the mission.', mission_file)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
malmoutils.parse_command_line(agent_host)
my_clients = MalmoPython.ClientPool()
my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
agentID = 0
################################################

############################### prepare training
action_list = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
ckpt_dir = os.path.abspath("./checkpoints")
ckpt_save_rate = 50
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)
bs = 2000
update_rate=10
dqn = DQN(batch_size=bs, update_rate=update_rate, lr=2e-4)
memory = []
mem_size = 50000
expID = 0
mission_change_rate = 25
#################################################


# ################################## filling memory
# mission_xml_path = os.path.join(agent_host.getStringArgument('mission_file'), "Maze0.xml")
# world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
# curr_state = get_curr_state(world_state)
# done = False
# curr_pos = (None, None, None)
# while len(memory) < mem_size:
#     # change mission xml per 20 times
#     if len(memory) % mission_change_rate == 0:
#         mission_xml_path = get_random_mission_xml_path(agent_host)
#     if done:
#         logger.info("curr records in memory %d" % (len(memory)))
#         world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
#         curr_state = get_curr_state(world_state)
#     act = epsilon_greedy(dqn, curr_state, eps=1)
#     done, reward, world_state, pos = step(agent_host, action_list[act])
#     # if stay in same place and not ended, set reward to -10
#     if not done and curr_pos[0] == pos[0] and curr_pos[2] == pos[2]:
#         reward = -10.0
#     logger.info("- action=\"{}\", reward={}, pos={}".format(action_list[act], reward, pos))
#     next_state = get_next_state(world_state, curr_state)
#     memory.append(Transition(curr_state, act, reward, next_state, done))
#     curr_state = next_state
#     curr_pos = pos
# logger.info("Finished populating memory")
# #################################################


######################################## training
epochs = 4000
start_eps = 0.9
end_eps = 0.15
start_decay_epoch = 200
end_decay_epoch = 2000
n_batch = 32
done = False
losses = []
for i in range(epochs):
    if i % ckpt_save_rate == 0:
        save_ckpt(dqn.model_pred, "ckpt@epoch%d"%(i), ckpt_dir)
    logger.info("%d-th episode"%(i))
    if i % mission_change_rate == 0:
        mission_xml_path = get_random_mission_xml_path(agent_host)
    world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID, logger)
    curr_state = get_curr_state(world_state)
    curr_pos = (None, None, None)
    done = False
    curr_eps = get_epsilon(i, epochs)
    curr_reward = 0
    step_cnt = 0
    visited = set([])
    while not done:
        # step
        step_cnt += 1
        act = epsilon_greedy(dqn, curr_state, curr_eps)
        done, reward, world_state, pos = step(agent_host, action_list[act])
        next_state = get_next_state(world_state, curr_state)
        # if stay in same place and not ended, set reward to -10
        if not done and curr_pos[0] == pos[0] and curr_pos[2] == pos[2]:
            reward = -10.0
        visited.add(pos2id(pos))
        if len(memory) >= mem_size:
            memory.pop(0)
        memory.append(Transition(curr_state, act, reward, next_state, done))
        curr_state = next_state
        curr_pos = pos
        curr_reward += reward
        logger.info("- step " + str(step_cnt) + " of epoch"+ str(i+1) +": action= "+action_list[act]+" , reward= "+str(reward))
    logger.info("total reward @ epoch %d is %d"%(i+1, curr_reward))
    # train
    if len(memory) >= bs:
        i += 1
        loss = 0
        for _ in tqdm(range(n_batch)):
            loss += dqn.train_once(memory)
        logger.info("loss after epoch {} is {}".format(i + 1, loss / n_batch))
        losses.append(loss / n_batch)
    else:
        logger.info("populating memory pool: {}/{}".format(len(memory), mem_size))
#################################################


save_ckpt(dqn.model_pred, "ckpt@finished", ckpt_dir)
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("./logs/loss-{}.png".format(timestamp))
plt.show()