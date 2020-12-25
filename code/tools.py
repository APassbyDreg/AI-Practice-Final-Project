import json
import time
import os
import logging
from collections import namedtuple

import numpy as np


# nonlocal MalmoPython
try:
    from malmo import MalmoPython
except:
    import MalmoPython


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
MAX_RETRIES = 100
BLOCK_2_ID = {"carpet": 0,
              "wooden_slab": 0,
              "sea_lantern": 1,
              "fire": 2,
              "emerald_block": 3,
              "beacon": 4,
              "human": 5,
              "glass": 6,
              "netherrack": 7,
              "__else": 8}
WALL_BLOCKS = {"glass", "beacon", "sea_lantern", "fire"}
ROAD_BLOCKS = {"netherrack", "wooden_slab", "carpet", "fire"}
SAFE_BLOCKS = {"netherrack", "wooden_slab", "carpet"}
EXIT_BLOCKS = {"emerald_block"}
FIRE_BLOCKS = {"fire"}
FLAMABLE_BLOCKS = {"wooden_slab", "carpet"}
STATE_LAYERS = [WALL_BLOCKS, ROAD_BLOCKS, EXIT_BLOCKS, SAFE_BLOCKS, FIRE_BLOCKS, FLAMABLE_BLOCKS]

def grid_process(world_state):
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor10x10', 0)
    layers = []
    for l in STATE_LAYERS:
        layer = [1 if b in l else 0 for b in grid]
        layers.append(np.array(layer))
    full_grid = np.stack(layers)
    full_grid = full_grid.reshape([len(layers), 13, 13])
    return full_grid


def get_epsilon(curr_step, decay_end_step, decay_start_step=0, start_eps=1, end_eps=0.1):
    assert(decay_start_step < decay_end_step)
    curr_step = min(curr_step, decay_end_step)
    pos = max(0, curr_step - decay_start_step)
    return start_eps + (end_eps - start_eps) * (pos) / (decay_end_step - decay_start_step)


def epsilon_greedy(estimator, obs, eps, num_actions=4, logger=None):
    action_probs = np.ones(num_actions, dtype=float) * eps / num_actions
    q_values = estimator(np.expand_dims(obs, 0)).detach().numpy()
    best_action = np.argmax(q_values[0])
    if logger is not None:
        logger.info(f"q_values: {q_values}")
    action_probs[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(num_actions), p=action_probs)
    return action, best_action


def get_random_mission_xml_path(agent_host, n_maze=10):
    mission_file = agent_host.getStringArgument('mission_file')
    mazeNum = np.random.randint(0, n_maze)
    mission_file = os.path.join(mission_file,"Maze%s.xml"%mazeNum)
    return mission_file


def reset_world(agent_host, 
                mission_xml_path,
                my_clients,
                agentID,
                expID,
                logger):
    reseted = False
    while not reseted:
        my_mission_record = MalmoPython.MissionRecordSpec()
        with open(mission_xml_path, 'r') as f:
            logger.info("Loading mission from %s" % mission_xml_path)
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
                    logger.info(f"Error starting mission: {e}")
                    exit(1)
                else:
                    time.sleep(2.5)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        reseted = len(world_state.observations) > 0
    return world_state


def get_curr_state(world_state):
    state = grid_process(world_state)
    # state = np.stack([state] * 8, axis=0)
    return state


def get_next_state(world_state, curr_state):
    next_state = grid_process(world_state)
    # next_state = np.append(curr_state[1:, :, :], np.expand_dims(next_state, 0), axis=0)
    return next_state


def step(agent_host, cmd):
    agent_host.sendCommand(cmd)
    time.sleep(0.1)
    world_state = agent_host.peekWorldState()
    num_frames_seen = world_state.number_of_video_frames_since_last_state

    while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen or len(world_state.observations) == 0:
        time.sleep(0.1)
        world_state = agent_host.peekWorldState()
    
    done = not world_state.is_mission_running
    try:
        reward = world_state.rewards[-1].getValue()
    except:
        reward = -10
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    pos = (observations['XPos'], observations['YPos'], observations['ZPos'])
    return done, reward, world_state, pos


def pos2id(pos):
    x = pos[0]
    z = pos[2]
    return int(x*10) * 2333 + int(z*10)