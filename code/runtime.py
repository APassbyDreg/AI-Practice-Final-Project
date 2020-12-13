import shutil
import torch
import json
import pandas as pd
import numpy as np
import os

from tqdm import tqdm


try:
    from malmo import MalmoPython
except:
    import MalmoPython

try:
    from code.models import *
    from code.tools import *
    from source import malmoutils
except:
    from tools import *
    from models import *
    import malmoutils


################################## prepare malmo
agent_host = MalmoPython.AgentHost()
mission_file = agent_host.getStringArgument('mission_file')
mission_file = os.path.join(mission_file, "Maze0.xml")
currentMission = mission_file
schema_dir = None
try:
    schema_dir = "mazes"
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(schema_dir)
if not os.path.exists(mission_file):
    print("Could not find Maze.xml under MALMO_XSD_PATH")
    exit(1)
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
ckpt_dir = os.path.abspath("./checkpoints")
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)
estimator = Estimator()
predictor = Estimator()
memory = []
mem_size = 1000
expID = 0
epochs = 1000
start_eps = 1
end_eps = 0.1
decay_point = 200
#################################################


################################## filling memory
for i in range(mem_size):
    mission_xml_path = get_random_mission_xml_path(agent_host)
    my_mission_record = malmoutils.get_default_recording_object(agent_host, f"save_{expID}-rep{i}")
    world_state = reset_world(agent_host, mission_xml_path, my_clients, my_mission_record, agentID, expID)

    curr_state = get_curr_state(world_state)
    eps = start_eps
    
