import json
import time
import os
from collections import namedtuple

import numpy as np


# nonlocal MalmoPython
try:
    from malmo import MalmoPython
except:
    import MalmoPython


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
MAX_RETRIES = 5



def grid_process(world_state):
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor10x10', 0)
    block2id = {"carpet": 0,
                "sea_lantern": 1,
                "fire": 2,
                "emerald_block": 3,
                "beacon": 4,
                "air": 5}
    for i, block in enumerate(grid):
        if block in block2id.keys():
            grid[i] = block2id[block]
        else:
            grid[i] = -1
    full_grid = np.reshape(np.array(grid), [13, 13])
    return full_grid[2:-2, 2:-2]


def get_epsilon(curr_step, total_step, start_eps=1, end_eps=0.1, decay_start_step=0):
    assert(decay_start_step < total_step)
    pos = max(0, curr_step-decay_start_step)
    return end_eps + (start_eps - end_eps) * (pos) / (total_step - decay_start_step)


def epsilon_greedy(estimator, obs, eps, num_actions):
    action_probs = np.ones(num_actions, dtype=float) * eps / num_actions
    q_values = estimator(np.expand_dims(obs, 0))
    best_action = np.argmax(q_values.detach().numpy())
    action_probs[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(num_actions), p=action_probs)
    return action


def get_random_mission_xml_path(agent_host):
    mission_file = agent_host.getStringArgument('mission_file')
    mazeNum = np.random.randint(0, 4)
    mission_file = os.path.join(mission_file,"Maze%s.xml"%mazeNum)
    return mission_file


def reset_world(agent_host, 
                mission_xml_path,
                my_clients,
                agentID,
                expID):
    my_mission_record = MalmoPython.MissionRecordSpec()
    with open(mission_xml_path, 'r') as f:
        print("Loading mission from %s" % mission_xml_path)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.setViewpoint(2)

    for retry in range(MAX_RETRIES):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s" % (expID))
            break
        except RuntimeError as e:
            if retry == MAX_RETRIES - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2.5)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
    agent_host.sendCommand("look -1")
    agent_host.sendCommand("look -1")
    while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
        world_state = agent_host.peekWorldState()

    return world_state


def get_curr_state(world_state):
    state = grid_process(world_state)
    state = np.stack([state] * 4, axis=0)
    return state


def get_next_state(world_state, curr_state):
    next_state = grid_process(world_state)
    next_state = np.append(curr_state[1:, :, :], np.expand_dims(next_state, 0), axis=0)
    return next_state


def step(agent_host, cmd):
    agent_host.sendCommand(cmd)
    world_state = agent_host.peekWorldState()
    num_frames_seen = world_state.number_of_video_frames_since_last_state

    while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
        world_state = agent_host.peekWorldState()
    
    done = not world_state.is_mission_running
    try:
        reward = world_state.rewards[-1].getValue()
    except:
        reward = -10
    return done, reward, world_state
