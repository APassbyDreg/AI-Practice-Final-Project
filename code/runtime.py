try:
    from malmo import MalmoPython
except:
    import MalmoPython

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


##################################### set logger
if not os.path.exists("./logs"):
    os.makedirs("./logs")
timestamp = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"./logs/{timestamp}.log")
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
################################################


################################## prepare malmo
agent_host = MalmoPython.AgentHost()
mission_file_path = "/home/apd/MalmoPlatform/Schemas"
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
                                     'Path/to/file from which to load the mission.', mission_file_path)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
malmoutils.parse_command_line(agent_host)
my_clients = MalmoPython.ClientPool()
my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))    
agentID = 0


############################### prepare training
action_list = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
ckpt_dir = os.path.abspath("./checkpoints")
ckpt_save_rate = 50
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)
#################################################


######################################## training
bs = 256
memory = []
mem_size = 2048
mission_change_rate = 1
n_maze = 10
num_epoch = 2000
start_eps = 1
end_eps = 0.4
start_decay_epoch = 0
end_decay_epoch = 1500
n_batch = 32
done = False
losses = []
i = 0
success = []
dqn = DQN(batch_size=bs, lr=1e-4)
while i < num_epoch:
    if i % ckpt_save_rate == 0:
        save_ckpt(dqn.model_pred, "ckpt@epoch{:04d}".format(i), ckpt_dir)
    if i % mission_change_rate == 0:
        mission_xml_path = get_random_mission_xml_path(agent_host, n_maze)
    world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, 0, logger)
    curr_state = get_curr_state(world_state)
    done = False
    curr_eps = get_epsilon(i, end_decay_epoch, start_decay_epoch, start_eps=start_eps, end_eps=end_eps)
    step_cnt = 0
    visited = set([])
    logger.info(f"{i}-th episode, eps={curr_eps}")
    while not done:
        # step
        step_cnt += 1
        act = epsilon_greedy(dqn, curr_state, curr_eps, logger=None)
        done, reward, world_state, pos = step(agent_host, action_list[act])
        next_state = get_next_state(world_state, curr_state)
        # if stay in same place and not ended, set reward to -10
        if not done and pos2id(pos) in visited:
            reward = -4.0
        visited.add(pos2id(pos))
        if len(memory) >= mem_size:
            memory.pop(0)
        memory.append(Transition(curr_state, act, reward, next_state, done))
        curr_state = next_state
        logger.info(f"- step {step_cnt} of epoch {i+1}: action=\"{action_list[act]}\", reward={reward}, pos={pos}")
    success.append(0 if reward <= 0 else 1)
    if len(success) > 50:
        logger.info(f"success rate of last 50 epoches is {sum(success[-50:])/50}")
    # train or save to memory
    if len(memory) >= bs:
        i += 1
        loss = 0
        for _ in tqdm(range(n_batch)):
            loss += dqn.train_once(memory)
        logger.info(f"loss after epoch {i+1} is {loss/n_batch}")
        losses.append(loss/n_batch)
    else:
        logger.info(f"populating memory pool: {len(memory)}/{mem_size}")
# save final ckpt
save_ckpt(dqn.model_pred, "ckpt@finished", ckpt_dir)
success_rate = [0]
for s in success:
    success_rate.append(s * 0.02 + success_rate[-1] * 0.98)
#################################################


######################################### testing
test_result = []
test_repeat = 25
for mazeNum in range(n_maze):
    test_result.append([])
    mission_file_path = agent_host.getStringArgument('mission_file')
    mission_xml_path = os.path.join(mission_file_path,"Maze%s.xml" % mazeNum)
    for repeat in range(test_repeat):
        world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, 0, logger)
        curr_state = get_curr_state(world_state)
        done = False
        step_cnt = 0
        while not done:
            # step
            step_cnt += 1
            act = epsilon_greedy(dqn, curr_state, eps=0, logger=None)
            done, reward, world_state, pos = step(agent_host, action_list[act])
            next_state = get_next_state(world_state, curr_state)
            curr_state = next_state
            logger.info(f"- step {step_cnt} of epoch {i+1}: action=\"{action_list[act]}\", reward={reward}, pos={pos}")
        test_result[-1].append(0 if reward <= 0 else 1)
        logger.info(f"test {repeat} on maze{mazeNum}: {test_result[-1][-1] == 1}")
    logger.info(f"success rate on maze{mazeNum} is {sum(test_result[-1]) / test_repeat}")
json.dump(test_result, open(f"./logs/test-result-{timestamp}.json", "w"))
#################################################


plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f"./logs/loss-{timestamp}.png")

plt.clf()
plt.plot(success_rate[1:])
plt.xlabel("epoch")
plt.ylabel("success_rate over ~ 50 epoches")
plt.savefig(f"./logs/success-rate-{timestamp}.png")
plt.show()