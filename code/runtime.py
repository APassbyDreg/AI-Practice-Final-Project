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


################################## prepare malmo
agent_host = MalmoPython.AgentHost()
mission_file = "/home/apd/MalmoPlatform/Schemas"
# schema_dir = None
# try:
#     schema_dir = "mazes"
# except KeyError:
#     print("MALMO_XSD_PATH not set? Check environment.")
#     exit(1)
# mission_file = os.path.abspath(schema_dir)
# if not os.path.exists(mission_file):
#     print("Could not find Maze.xml under MALMO_XSD_PATH")
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
mission_change_rate = 32
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)
dqn = DQN()
memory = []
mem_size = 2048
expID = 0
epochs = 400
start_eps = 0.9
end_eps = 0.1
#################################################


################################## filling memory
mission_xml_path = os.path.join(agent_host.getStringArgument('mission_file'), "Maze0.xml")
world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
curr_state = get_curr_state(world_state)
done = False
while len(memory) < mem_size:
    # change mission xml per 20 times
    if len(memory) % mission_change_rate == 0:
        mission_xml_path = get_random_mission_xml_path(agent_host)
    if done:
        world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
        curr_state = get_curr_state(world_state)
        print(f"curr records in memory {len(memory)}")
    act = epsilon_greedy(dqn, curr_state, eps=1)
    done, reward, world_state = step(agent_host, action_list[act])
    next_state = get_next_state(world_state, curr_state)
    memory.append(Transition(curr_state, act, reward, next_state, done))
    curr_state = next_state
print("Finished populating memory")
#################################################


######################################## training
n_batch = 16
mission_xml_path = os.path.join(agent_host.getStringArgument('mission_file'), "Maze0.xml")
world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
done = False
for i in range(epochs):
    if i % ckpt_save_rate == 0:
        save_ckpt(dqn.model_pred, "ckpt@epoch{:04d}".format(i), ckpt_dir)
    print(f"{i}-th episode")
    if i % mission_change_rate == 0:
        mission_xml_path = get_random_mission_xml_path(agent_host)
    world_state = reset_world(agent_host, mission_xml_path, my_clients, agentID, expID)
    curr_state = get_curr_state(world_state)
    done = False
    curr_eps = get_epsilon(i, epochs)
    curr_reward = 0
    step_cnt = 0
    while not done:
        # step
        step_cnt += 1
        act = epsilon_greedy(dqn, curr_state, curr_eps)
        done, reward, world_state = step(agent_host, action_list[act])
        next_state = get_next_state(world_state, curr_state)
        memory.pop(0)
        memory.append(Transition(curr_state, act, reward, next_state, done))
        curr_state = next_state
        curr_reward += reward
        print(f"- step {step_cnt} of epoch {i+1}: action=\"{action_list[act]}\", reward={reward}")
    print(f"total reward @ epoch{i+1} is {curr_reward}")
    # train
    loss = 0
    for _ in tqdm(range(n_batch)):
        loss += dqn.train_once(memory)
    print(f"loss after epoch {i+1} is {loss/n_batch}")
#################################################


save_ckpt(dqn.model_pred, "ckpt@finished", ckpt_dir)