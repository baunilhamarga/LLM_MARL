import pandas as pd

from gym_dragon.envs import MiniDragonEnv
from gym_dragon.core import Region, Agent, Tool, Bomb
from gym_dragon.wrappers import MiniObs
import openai
import time
import json
import os
import re
from collections.abc import Mapping
from numbers import Number
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

COLOR_TO_STR={0: 'Red',1:'Green',2:'Blue'}
# ACTION_TO_STR={1: 'inspect_bomb',7:'go_to_node_0',8:'go_to_node_3',9:'go_to_node_5',10:'go_to_node_6',11:'go_to_node_8'}
BOMBSTATE_TO_STR = {0: 'inactive',1: 'active',2: 'exploded',3: 'defused'}
openai.api_key = os.environ.get("OPENAI_API_KEY", "na")
PRESETS = {
    'village': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63), 11: (32, 56), 12: (41, 52), 14: (15, 67), 15: (21, 55), 16: (45, 52), 17: (5, 62), 20: (42, 60), 21: (6, 56), 22: (48, 52), 23: (37, 78), 24: (40, 74), 26: (46, 65), 27: (48, 60), 28: (11, 69), 31: (14, 80), 32: (46, 75), 33: (6, 69), 34: (20, 80), 36: (48, 70), 38: (21, 69), 40: (6, 76), 41: (23, 76), 43: (46, 80), 45: (28, 88), 46: (43, 91), 47: (29, 76), 48: (15, 92), 49: (9, 87), 51: (47, 85), 52: (4, 82), 53: (32, 69), 55: (34, 92), 56: (9, 82), 57: (31, 82), 58: (39, 89), 63: (46, 90), 64: (48, 90), 65: (27, 96), 66: (20, 96), 67: (44, 98), 68: (5, 95), 69: (11, 98), 70: (39, 98), 71: (34, 98), 72: (48, 98), 73: (34, 75), 75: (6, 98), 77: (6, 90)},
    'default': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63)},
    'easy': {0: (28, 56), 8: (22, 63), 6: (33, 64)},
    'medium': {23: (37, 78), 34: (20, 80), 38: (21, 69), 41: (23, 76), 47: (29, 76), 53: (32, 69), 57: (31, 82), 73: (34, 75)},
    'hard': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63), 14: (15, 67), 20: (42, 60), 23: (37, 78), 31: (14, 80), 34: (20, 80), 38: (21, 69), 41: (23, 76), 47: (29, 76), 53: (32, 69), 57: (31, 82), 73: (34, 75)},
}
class DragonTextEnv():
    def __init__(self,seed = None, include_agent_action = False,allow_comm = True,act_and_comm = True,tool_per_agent = 2, preset='default'):
        self.seed = seed

        self.valid_node = list(PRESETS[preset].keys())
        self.include_agent_action = include_agent_action
        self.allow_comm = allow_comm
        self.act_and_comm = act_and_comm
        self.tool_per_agent = tool_per_agent
        self.cutoff_activated = False
        self.round = 1

        self.env = MiniDragonEnv(mission_length = 999,
                        recon_phase_length=0,
                         include_chained_bombs=False,
                         include_fire_bombs=False,
                         include_fuse_bombs=False,
                         color_tools_only=True,
                         valid_nodes=PRESETS[preset],
                        obs_wrapper=MiniObs)
        self.env.seed(self.seed)

        if self.tool_per_agent == 2:
            self.env.reset(csv_path =None,
                           num_bombs_per_region = 5,
                           start_location = None,
                           start_regions=set(Region.village),
                           tool_allocation = {'alpha':{Tool.red:99,Tool.green:99},'bravo':{Tool.blue:99,Tool.green:99},'charlie':{Tool.red:99,Tool.blue:99}})
        else:
            self.env.reset(csv_path =None,
                           num_bombs_per_region = 5,
                           start_location = None,
                           start_regions=set(Region.village),
                           tool_allocation = {'alpha':{Tool.red:99},'bravo':{Tool.green:99},'charlie':{Tool.blue:99}})


    def step(self,agent_id, round, initial_actions, communications, tom_questions=None, tom_reasoning=False):
        # action is object
        # agent is str index
        valid_action = True
        reward = {agent_id: 0 for agent_id in self.env.agents.keys()}
        info = {agent_id: {} for agent_id in self.env.agents.keys()}
        prev_agent_health = {agent.id: agent.health for agent in self.env.agents.values()}
        self.round = round

        action = initial_actions[agent_id]
        obs_text = 'Your action is invalid.'

        Action = self.env.action_enum
        agent = self.env.agents[agent_id]
        if action is None:
            obs_text = "Invalid action."
            valid_action = False
        elif action == Action.inspect_bomb:
            if agent.bomb:
                bomb_id = str(agent.bomb.id)
                sequence = '-'.join([COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                num_stage = str(agent.bomb.num_stages)
                obs_text = "You inspected Bomb {bomb_id}. This bomb is a {num_stage}-stage bomb and its remaining sequence is {sequence}.".format(
                    bomb_id=bomb_id, num_stage=num_stage, sequence=sequence)

                agent.bomb.inspect()
                self.env._inspected_bombs[agent.id].add(agent.bomb)
                self.env.observations[agent.id].update_from_inspection()
            else:
                current_room = str(agent.node.id)
                obs_text = "There is no bomb in the current current location, Room {current_room}, for you to inspect.".format(
                    current_room=current_room)
                valid_action = False
        elif action.node() is not None:
            # Go to the given node
            if self.env._get_action_mask(agent_id)[action] or action.node().id == agent.node.id:
                room_id = str(action.node().id)
                room_agents = ' and '.join([str(x.id) for x in action.node().agents])
                room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in action.node().bombs])
                obs_text = "You moved to Room {room_id}. In the new room you found {room_agents}, {room_bombs}.".format(
                    room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

                agent.go_to(action.node())
            else:
                room_id = str(action.node().id)
                current_room = str(agent.node.id)
                obs_text = "You can not directly move to Room {room_id} because it is not adjacent to your current location, Room {current_room}. Consider taking a detour to another room first and then move to your destination.".format(
                    room_id=room_id, current_room=current_room)
                valid_action = False


        elif action.tool() is not None:
            tool = agent.get_tool_from_inventory(action.tool())
            if tool in Tool.bomb_tools():
                if agent.bomb:
                    if tool.color == agent.bomb.color:
                        agent.bomb.apply_tool(tool)
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = str(agent.bomb.id)
                        sequence = '-'.join(
                            [COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                        state = BOMBSTATE_TO_STR[agent.bomb.state]

                        if agent.bomb.state == Bomb.BombState.defused:
                            reward = agent.bomb.value
                            obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb is defused successfully.".format(
                                tool_color=tool_color, bomb_id=bomb_id)
                        else:
                            obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb is {state} and its remaining sequence is {sequence}.".format(
                                tool_color=tool_color, bomb_id=bomb_id, state=state, sequence=sequence)

                    else:
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = agent.bomb.id
                        sequence = agent.bomb._full_sequence[agent.bomb._current_step:]
                        state = agent.bomb.state

                        obs_text = "You can not apply Tool {tool_color} to Bomb {bomb_id} because the sequence of this bomb is {sequence}. You will need to apply other color tool first.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)
                        valid_action = False
                else:
                    obs_text = "There is no bomb in your current location, room {room_id}, for you to defuse.".format(
                        room_id=agent.node.id)
                    valid_action = False
            else:
                obs_text = "You do not have {tool}. Consider asking your teammates who have this tool to help you defuse the bomb.".format(
                    tool=COLOR_TO_STR[action.tool()])
                valid_action = False


        self.env.tick()


        for agent_id in info:
            info[agent_id]['score'] = self.env.score

        obs, reward, done, info = self.env._get_obs(), reward, self.env._get_done(), info

        # team_location_text = 'Player alpha is in Room {loc_a}; Player bravo is in Room {loc_b}; Player charlie is in Room {loc_c}.'.format(
        #     loc_a=str(self.env.agents['alpha'].node.id), loc_b=str(self.env.agents['bravo'].node.id),
        #     loc_c=str(self.env.agents['charlie'].node.id))

        room_id = str(agent.node.id)
        room_agents = ' and '.join([str(x.id) for x in agent.node.agents])
        room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in agent.node.bombs])
        room_contents = "You are currently in Room {room_id}. Contents of this room include {room_agents}, {room_bombs}.".format(
            room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

        text = 'Your observation is: '
        text += 'Round: {timestep}. '.format(timestep=str(round+1))
        text += 'Total team score: {score}. '.format(score=str(self.env.score))
        text += f"Previous action results ({action.name.replace('_', ' ') if action != None else 'Invalid'}): " + obs_text + ' '
        text += 'Room contents: '+ room_contents+ ' '
        text += 'Communication is available this round. ' if not self.cutoff_activated else 'Communication is not available this round.'
        # text += 'Teammate Locations: ' + team_location_text + ' '



        if self.include_agent_action:
            text += 'Your teammates action in last round: '
            for a in initial_actions.keys():
                if initial_actions[a].name == 'remove_bomb_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Sent a communication message to the Team.')
                elif initial_actions[a].name == 'remove_help_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Invalid action.')
                else:
                    text += 'Player {id}: "{action}". '.format(action.name.replace('_', ' ') if action != None else 'Invalid')

        if self.allow_comm:
            text += 'Communication messages sent by your teammates: '
            for a in communications.keys():
                text += 'Player {id}: "{comm}". '.format(id=a, comm=communications[a])

            if tom_questions is not None and not self.cutoff_activated and tom_reasoning:
                text += 'Before sending your message to the team, consider the following questions (do not answer them): '
                for i, q in enumerate(tom_questions):
                    if q is not None:
                        text += '{index}: "{question}". '.format(index=i+1, question=q)

        # print(text)
        return obs, reward, done, info, text, valid_action, obs_text

    def   step_text(self,agent_id, round, initial_actions, communications, results=''):
        # action is object
        # agent is str index


        action = initial_actions[agent_id]
        obs_text = 'Your action is invalid.'

        Action = self.env.action_enum
        agent = self.env.agents[agent_id]
        if action is None:
            obs_text = "Invalid action."
        elif action == Action.inspect_bomb:
            if agent.bomb:
                bomb_id = str(agent.bomb.id)
                sequence = '-'.join([COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                num_stage = str(agent.bomb.num_stages)
                obs_text = "You inspected Bomb {bomb_id}. This bomb is a {num_stage}-stage bomb and its remaining sequence is {sequence}.".format(
                    bomb_id=bomb_id, num_stage=num_stage, sequence=sequence)

            else:
                current_room = str(agent.node.id)
                obs_text = "There is no bomb in the current current location, Room {current_room}, for you to inspect.".format(
                    current_room=current_room)
        elif action.node() is not None:
            # Go to the given node
            if self.env._get_action_mask(agent_id)[action] or action.node().id == agent.node.id:
                room_id = str(action.node().id)
                room_agents = ' and '.join([str(x.id) for x in action.node().agents])
                room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in action.node().bombs])
                obs_text = "You moved to Room {room_id}. In the new room you found {room_agents}, {room_bombs}.".format(
                    room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

            else:
                room_id = str(action.node().id)
                current_room = str(agent.node.id)
                obs_text = "You can not directly move to Room {room_id} because it is not adjacent to your current location, Room {current_room}. Consider taking a detour to another room first and then move to your destination.".format(
                    room_id=room_id, current_room=current_room)


        elif action.tool() is not None:
            tool = agent.get_tool_from_inventory(action.tool())
            if tool in Tool.bomb_tools():
                if agent.bomb:
                    if tool.color == agent.bomb.color:

                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = str(agent.bomb.id)
                        sequence = '-'.join(
                            [COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step+1:]])

                        obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb has a remaining sequence of {sequence}.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)

                    else:
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = agent.bomb.id
                        sequence = agent.bomb._full_sequence[agent.bomb._current_step:]


                        obs_text = "You can not apply Tool {tool_color} to Bomb {bomb_id} because the sequence of this bomb is {sequence}. You will need to apply other color tool first.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)
                else:
                    obs_text = "There is no bomb in your current location, room {room_id}, for you to defuse.".format(
                        room_id=agent.node.id)
            else:
                obs_text = "You do not have {tool}. Consider asking your teammates who have this tool to help you defuse the bomb.".format(
                    tool=COLOR_TO_STR[action.tool()])





        # team_location_text = 'Player alpha is in Room {loc_a}; Player bravo is in Room {loc_b}; Player charlie is in Room {loc_c}.'.format(
        #     loc_a=str(self.env.agents['alpha'].node.id), loc_b=str(self.env.agents['bravo'].node.id),
        #     loc_c=str(self.env.agents['charlie'].node.id))

        room_id = str(agent.node.id)
        room_agents = ' and '.join([str(x.id) for x in agent.node.agents])
        room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in agent.node.bombs])
        room_contents = "You are currently in Room {room_id}. Contents of this room include {room_agents}, {room_bombs}.".format(
            room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

        text = 'Your observation is: '
        text += 'Round: {timestep}. '.format(timestep=str(round+1))
        text += 'Total team score: {score}. '.format(score=str(self.env.score))
        text += f"Previous action results ({action.name.replace('_', ' ') if action != None else 'Invalid'}): " + results + ' '
        text += 'Room contents: '+ room_contents+ ' '
        text += 'Communication is available this round. ' if not self.cutoff_activated else 'Communication is not available this round.'
        # text += 'Teammate Locations: ' + team_location_text + ' '



        if self.include_agent_action:
            text += 'Your teammates action in last round: '
            for a in initial_actions.keys():
                if initial_actions[a].name == 'remove_bomb_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Sent a communication message to the Team.')
                elif initial_actions[a].name == 'remove_help_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Invalid action.')
                else:
                    text += 'Player {id}: "{action}". '.format(id=a, action=action.name.replace('_', ' ') if action != None else 'Invalid')

        if self.allow_comm:
            text += 'Communication messages sent by your teammates: '
            for a in communications.keys():
                text += 'Player {id}: "{comm}". '.format(id=a, comm=communications[a])



        # print(text)
        return text

    def decode_action(self, chat_output):
        lower = chat_output.lower().replace("“", '"').replace("”", '"')
        if 'action selection' in lower:
            lower = lower.split('action selection', 1)[1]  # keep text to the right of 'action selection'
        
        Action = self.env.action_enum
        comm = ''
        action = None
        if self.act_and_comm:
            if len(lower.split('message to team:')) > 1 and len(lower.split('"')) > 1:
                comm = lower.split('message to team:')[1].split('"')[1]
                lower = lower.split('message to team:')[0]+ lower.split('message to team:')[1].split('"')[2] if len(lower.split('message to team:')[1])>2 else ""
            else:
                comm = ''
            if self.cutoff_activated:
                comm = 'Not available due to communication cutoff.'
            tokens = ''.join(ch if ch.isalpha() else ' ' for ch in lower).split()  # tokenisation by spaces and removing non-alpha characters
            if 'inspect' in lower:
                action = Action.inspect_bomb
            elif 'move to room' in lower:
                m = re.search(r'\bmove to room\s*(\d+)\b', lower)
                if m:
                    room_id = int(m.group(1))
                if room_id in self.valid_node:
                    action = Action.go_to(room_id)
                else:
                    action = None
            elif 'go_to_node_' in lower:
                m = re.search(r'go_to_node_(\d+)', lower)
                if m:
                    room_id = int(m.group(1))
            elif 'apply' in lower or 'defuse' in lower:
                colour = next((tok for tok in tokens if tok in ('red', 'blue', 'green')), None)  # find the first colour token
                if colour == 'red':
                    action = Action.use_tool(Tool.red)
                elif colour == 'blue':
                    action = Action.use_tool(Tool.blue)
                elif colour == 'green':
                    action = Action.use_tool(Tool.green)
                else:
                    action = None 
            # elif 'wait' in chat_output.lower() or 'stay' in chat_output.lower():
            #     action = Action.remove_help_beacon
            else:
                action = None
                # action, comm = self.decode_action_API(chat_output, comm = comm)

        return action, comm






    def load(self,saved_files,ending_round=999):
        Action = self.env.action_enum
        # saved_files = {'alpha': DATA_PATH + 'gpt-4_0.7_alpha_05-25-19-18-01.json',
        #                'bravo': DATA_PATH + 'gpt-4_0.7_bravo_05-25-19-19-18.json',
        #                'charlie': DATA_PATH + 'gpt-4_0.7_charlie_05-25-19-20-49.json'}

        chat_agents = {'alpha': ChatAgent(agent_id='alpha'), 'bravo': ChatAgent(agent_id='bravo'),
                       'charlie': ChatAgent(agent_id='charlie')}
        initial_actions = {'alpha': Action.go_to(0), 'bravo': Action.go_to(0), 'charlie': Action.go_to(0)}
        communications = {'alpha': 'None', 'bravo': 'None', 'charlie': 'None'}

        round = 1
        record = saved_files['record']
        with open(record, 'r', encoding='utf-8') as f:
            data = f.read()
            new_data = data.replace('}{', '},{')
            json_data = json.loads(f'[{new_data}]')
            print(json_data)


        for r in json_data:
            if 'action' in r.keys():
                agent_id = r['agent_id']
                initial_actions[agent_id] = Action[r['action'].replace(' ', '_')]
                communications[agent_id] = r['comm']
                _, reward, done, info, obs_text, valid_action = self.step(agent_id, round, initial_actions, communications)
                round = r['round']
            else:
                agent_id = r['agent_id']
                initial_actions[agent_id], communications[agent_id] = self.decode_action(r["chat_output"])
                _, reward, done, info, obs_text, valid_action = self.step(agent_id, round, initial_actions, communications)
                round = r['round']
            if round >= ending_round:
                break



        for agent_id in ['alpha','bravo','charlie']:

            saved_file = saved_files[agent_id]
            with open(saved_file, 'r', encoding='utf-8') as f:
                file = json.load(f)
            chat_agents[agent_id] = ChatAgent(agent_id=file['agent_id'],model=file['model'],temperature=file['temperature'],message_history=file['message_history'],belief=True,allow_comm=True)

            initial_actions[agent_id],communications[agent_id] = self.decode_action(file['message_history'][-2]['content'])

        round += 1

        return chat_agents, initial_actions, communications, round


    def to_csv(self,saved_file,output_path):
        Action = self.env.action_enum
        initial_actions = {'alpha': Action.go_to(0), 'bravo': Action.go_to(0), 'charlie': Action.go_to(0)}
        communications = {'alpha': 'None', 'bravo': 'None', 'charlie': 'None'}

        output = []
        record = saved_file
        with open(record, 'r', encoding='utf-8') as f:
            data = f.read()
            new_data = data.replace('}{', '},{')
            json_data = json.loads(f'[{new_data}]')
            print(json_data)

        for r in json_data:

            agent_id = r['agent_id']
            initial_actions[agent_id], communications[agent_id] = self.decode_action(r["chat_output"])
            _, reward, done, info, obs_text = self.step(agent_id, initial_actions, communications)
            round = r['round']
            row = [round,agent_id,initial_actions[agent_id],communications[agent_id],reward,done,obs_text]
            output.append(row)

        csv = pd.DataFrame(output, columns=["round","agent_id","action","comm","reward","done","obs_text"])
        csv.to_csv(output_path)



MAX_RETRIES = 10
RETRY_DELAY = 3


MAP_DEFAULT = "0 : 3 5 6 8 \n\
3 : 0 8 \n\
5 : 0 6 \n\
6 : 0 5 8 \n\
8 : 0 3 6 \n\ "

MAP_NODES_DEFAULT = "0 : \n\
3 : \n\
5 : \n\
6 : \n\
8 : \n\ "

MAP_EASY = "0 : 6 8 \n\
6 : 0 8 \n\
8 : 0 6 \n\ "

MAP_NODES_EASY = "0 : \n\
6 : \n\
8 : \n\ "

MAP_MEDIUM = "23 : 47 73 \n\
34 : 38 41 57 \n\
38 : 34 53 \n\
41 : 34 47 \n\
47 : 23 41 57 73 \n\
53 : 38 73 \n\
57 : 34 47 \n\
73 : 23 47 53 \n\ "

MAP_NODES_MEDIUM = "23 : \n\
34 : \n\
38 : \n\
41 : \n\
47 : \n\
53 : \n\
57 : \n\
73 : \n\ "

MAP_HARD = "0 : 3 5 6 8 \n\
3 : 0 8 14 \n\
5 : 0 6 \n\
6 : 0 5 8 20 23 \n\
8 : 0 3 6 14 \n\
14 : 3 8 31 34 \n\
20 : 6 \n\
23 : 6 47 73 \n\
31 : 14 34 \n\
34 : 14 31 38 41 57 \n\
38 : 34 53 \n\
41 : 34 47 \n\
47 : 23 41 57 73 \n\
53 : 38 73 \n\
57 : 34 47 \n\
73 : 23 47 53 \n\ "

MAP_NODES_HARD = "0 : \n\
3 : \n\
5 : \n\
6 : \n\
8 : \n\
14 : \n\
20 : \n\
23 : \n\
31 : \n\
34 : \n\
38 : \n\
41 : \n\
47 : \n\
53 : \n\
57 : \n\
73 : \n\ "

MAP_VILLAGE = "0 : 3 5 6 8 \n\
3 : 0 8 14 15 17 21 \n\
5 : 0 6 12 16 22 27 \n\
6 : 0 5 8 11 20 23 24 \n\
8 : 0 3 6 14 \n\
11 : 6 \n\
12 : 5 \n\
14 : 3 8 17 28 31 34 \n\
15 : 3 \n\
16 : 5 \n\
17 : 3 14 21 40 \n\
20 : 6 24 26 27 \n\
21 : 3 17 \n\
22 : 5 \n\
23 : 6 24 45 47 73 \n\
24 : 6 20 23 26 32 46 \n\
26 : 20 24 32 36 \n\
27 : 5 20 \n\
28 : 14 33 \n\
31 : 14 34 40 48 49 \n\
32 : 24 26 43 \n\
33 : 28 \n\
34 : 14 31 38 41 57 \n\
36 : 26 \n\
38 : 34 53 \n\
40 : 17 31 52 \n\
41 : 34 47 \n\
43 : 32 51 \n\
45 : 23 48 55 65 \n\
46 : 24 58 67 70 72 \n\
47 : 23 41 57 73 \n\
48 : 31 45 49 66 68 69 \n\
49 : 31 48 56 \n\
51 : 43 63 64 \n\
52 : 40 \n\
53 : 38 73 \n\
55 : 45 \n\
56 : 49 \n\
57 : 34 47 \n\
58 : 46 \n\
63 : 51 \n\
64 : 51 \n\
65 : 45 71 \n\
66 : 48 \n\
67 : 46 \n\
68 : 48 75 77 \n\
69 : 48 \n\
70 : 46 \n\
71 : 65 \n\
72 : 46 \n\
73 : 23 47 53 \n\
75 : 68 \n\
77 : 68 \n\ "

MAP_NODES_VILLAGE = "0 : \n\
3 : \n\
5 : \n\
6 : \n\
8 : \n\
11 : \n\
12 : \n\
14 : \n\
15 : \n\
16 : \n\
17 : \n\
20 : \n\
21 : \n\
22 : \n\
23 : \n\
24 : \n\
26 : \n\
27 : \n\
28 : \n\
31 : \n\
32 : \n\
33 : \n\
34 : \n\
36 : \n\
38 : \n\
40 : \n\
41 : \n\
43 : \n\
45 : \n\
46 : \n\
47 : \n\
48 : \n\
49 : \n\
51 : \n\
52 : \n\
53 : \n\
55 : \n\
56 : \n\
57 : \n\
58 : \n\
63 : \n\
64 : \n\
65 : \n\
66 : \n\
67 : \n\
68 : \n\
69 : \n\
70 : \n\
71 : \n\
72 : \n\
73 : \n\
75 : \n\
77 : \n\ "

PRESET_MAPS = {
    'village': (MAP_VILLAGE, MAP_NODES_VILLAGE),
    'default': (MAP_DEFAULT, MAP_NODES_DEFAULT),
    'easy': (MAP_EASY, MAP_NODES_EASY),
    'medium': (MAP_MEDIUM, MAP_NODES_MEDIUM),
    'hard': (MAP_HARD, MAP_NODES_HARD),
}

BACKGROUND_PROMPT_NEW = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. Your call sign is {agent_id}\
The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6. Room 3 is linked to room 8. And room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms at each turn.\
The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Your task is to maximize the team score as soon as possible. The challenge is that the bomb locations and sequences are unknown to players at the start.\
Tools: Each player is equipped with two color-coded wire cutters. Player Alpha has red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red.\
Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, or 3) Apply your wire cutters to a bomb in the current room. \
Communications: In addition to selecting an action to take from the above list, you can also send communication message texts to both of your teammates in each round. The message text you sent will be shared with both of your teammates in their observation in the next round. \
Observation: While you can only see what's in your current room and read text messages from teammates. You'll also be informed of the current round number, team score and the current location of your teammates. Your teammates have the same observability as you. They will not be able to know your action and its consequences unless you explicitly communicate. \
You will be playing as Player {agent_id}. To facilitate our interaction, reply your action selection and communication messages in this fixed format: Action selection: Your action. Message to Team: “Your Message”. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. Are you ready to take on this explosive challenge?"

BACKGROUND_PROMPT_NOCOMM = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. Your call sign is {agent_id}\
The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6. Room 3 is linked to room 8. And room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms at each turn.\
The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Your task is to maximize the team score as soon as possible. The challenge is that the bomb locations and sequences are unknown to players at the start.\
Tools: Each player is equipped with two color-coded wire cutters. Player Alpha has red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red.\
Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, or 3) Apply your wire cutters to a bomb in the current room. \
Observation: While you can only see what's in your current room. You'll also be informed of the current round number and team score. Your teammates have the same observability as you. They will not be able to know your action and its consequences unless you are in the same room. \
You will be playing as Player {agent_id}. To facilitate our interaction, reply your action selection in this fixed format: Action selection: Your action. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. Are you ready to take on this explosive challenge?"

BACKGROUND_PROMPT_CUTOFF = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. Your call sign is {agent_id}\
The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6. Room 3 is linked to room 8. And room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms at each turn.\
The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Your task is to maximize the team score as soon as possible. The challenge is that the bomb locations and sequences are unknown to players at the start.\
Tools: Each player is equipped with two color-coded wire cutters. Player Alpha has red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red.\
Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, or 3) Apply your wire cutters to a bomb in the current room. \
Communications: In addition to selecting an action to take from the above list, you can also send communication message texts to both of your teammates in each round. The message text you sent will be shared with both of your teammates in their observation in the next round. \
Observation: While you can only see what's in your current room and read text messages from teammates. You'll also be informed of the current round number, team score and the current location of your teammates. Your teammates have the same observability as you. They will not be able to know your action and its consequences unless you explicitly communicate. \
Extra challenge: at each round, there is a chance to lose communication with your teamates for the duration of the entire round. You will be informed with the message 'Communication cutoff: no communication available this round'. In this scenario, you should choose your next action based on all previous information and, when communication is available again in the next round, you should add to your regular team message any relevant information that your teamates might have missed during the cutoff. \
You will be playing as Player {agent_id}. To facilitate our interaction, reply your action selection and communication messages in this fixed format: Action selection: Your action. Message to Team: “Your Message”. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. Are you ready to take on this explosive challenge?"

# BACKGROUND_PROMPT = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6, room 3 is linked to room 8, and room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms each turn. The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. The challenge is that the bomb locations and sequences are unknown to players at the start. Your Tools: Each player is equipped with two color-coded wire cutters. As player Alpha, you have red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red. Your Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, 3) Apply your wire cutters to a bomb, or 4) send text messages to both of your teammates. Observation: While you can only see what's in your current room and read text messages from teammates. You'll also be informed of the current round number. Any successful bomb defusal, along with the accumulated team score, is broadcasted to all players. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Remember, your actions must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. You will be playing as Player {agent_id}. Are you ready to take on this explosive challenge?"
# INSTRUCT_PROMPT = "Your current observation is: Round: 1. Total team score: 0. Observation: You are currently in Room 0 with both of your teammates. In the room you also found bomb 1 with unknown sequence. There is no other bomb in the current room. Teammate Locations: Player alpha is in Room 0; Player bravo is in Room 0; Player charlie is in Room 0. Communication messages sent by your teammates: Player alpha: "". Player bravo: "". Player charlie: "". Given the above observation, what is your next action?"

INITIAL_BELIEF = "Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. Your role: You are playing as Player {agent_id}.\n Current round: 0.\n Total team score: 0.\n Restuls: You moved to Room {initial_node}. In the new room you found Player alpha, Player bravo, Player charlie, and Bomb {initial_bomb}.\n Room Contents: You are currently in Room {initial_node} with both of your teammates. In the room you also found bomb {initial_bomb} with unknown sequence. There is no other bomb in the current room.\n Teammate Locations: Player alpha is in Room {initial_node}; Player bravo is in Room {initial_node}; Player charlie is in Room {initial_node}.\n Room connectivity: Room 0 is connected to room 3, 5, 6,8. Room 3 is connected to room 0. Room 5 is connected to room 0 and 6. Room 6 is connected to room 0 and 8. Room 8 is connected to room 0 and 6.\n Bomb Intel: Bomb {initial_bomb}: Located in Room {initial_node}. The phase sequence is Unknown. Ohter bomb details currently unknown.\n Communication messages sent by your teammates: Player alpha: None. Player bravo: None. Player charlie: None.\n Tool inventory: Alpha: Equipped with red and green wire cutters. Bravo: Equipped with green and blue wire cutters. Charlie: Equipped with red and blue wire cutters.\n Available action options: 1. To move to an adjacent room, say: 'Move to Room X'. 2. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. 3. To apply a wire cutter tool, say: 'Apply X Tool'. 4. To send a message to your teammates, say: 'Message to Team: \"Your Message\"'."

INITIAL_BELIEF_NOCOMM = "Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. Your role: You are playing as Player {agent_id}.\n Current round: 0.\n Total team score: 0.\n Restuls: You moved to Room {initial_node}. In the new room you found Player alpha, Player bravo, Player charlie, and Bomb {initial_bomb}.\n Room Contents: You are currently in Room {initial_node} with both of your teammates. In the room you also found bomb {initial_bomb} with unknown sequence. There is no other bomb in the current room.\n Teammate Locations: Player alpha is in Room {initial_node}; Player bravo is in Room {initial_node}; Player charlie is in Room {initial_node}.\n Room connectivity: Room 0 is connected to room 3, 5, 6,8. Room 3 is connected to room 0. Room 5 is connected to room 0 and 6. Room 6 is connected to room 0 and 8. Room 8 is connected to room 0 and 6.\n Bomb Intel: Bomb {initial_bomb}: Located in Room {initial_node}. The phase sequence is Unknown. Ohter bomb details currently unknown.\n Tool inventory: Alpha: Equipped with red and green wire cutters. Bravo: Equipped with green and blue wire cutters. Charlie: Equipped with red and blue wire cutters.\n Available action options: 1. To move to an adjacent room, say: 'Move to Room X'. 2. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. 3. To apply a wire cutter tool, say: 'Apply X Tool'. 4."



UPDATE_PROMPT = "Please update your belief state based on the above observation, and send me your results in the same format as previously. DO NOT add additional explanations. Return your answer begining with: 'Below is your current belief'"


INITIAL_PROMPT = "Given the above belief state, {agent_id}, what is your next action?"


CUTOFF = 'Communication cutoff: no communication available this round'


BACKGROUND_PROMPT_NEW_IMPROVED = "Welcome to our interactive text game! You are a specialist on a three-person search-and-rescue team. Your call sign is {agent_id}.\n\n----------------  ROLE AND OBJECTIVE  ----------------\n* Defuse all 5 hidden bombs as fast as possible.\n* A team earns 10 x (number of phases) points for each bomb defused.\n* Maximize the final team score.\n\n----------------  ENVIRONMENT MAP  ----------------  \nThe facility is a connected graph of rooms labeled with integers:\n{map}\nYou may move only to rooms listed as directly connected to your current location.\n\n----------------  BOMB TYPES  ----------------\n* Bombs may have 1, 2, or 3 color phases. Defuse them by applying the matching colored tools in order.\n\n----------------  TOOLS PER PLAYER  ----------------\n* Alpha   - red, green\n* Bravo   - green, blue\n* Charlie - blue, red\n\n----------------  VALID ACTIONS  ----------------\n1. Move to an adjacent room     -> Move to Room X\n2. Inspect a bomb's phase list  -> Inspect Bomb\n3. Apply a wire cutter tool     -> Apply <Color> Tool\n\n----------------  COMMUNICATION  ----------------\n* Each round you may append ONE message to teammates (they will read it next round).\n\n----------------  OBSERVATION INFO  ----------------  \nAt the start of every round you learn:\n* Current round number and cumulative team score\n* Your room contents (bombs, players)\n* Locations of teammates\n* Messages sent in the previous round\n\n----------------  REPLY FORMAT  ----------------  \nTo facilitate our interaction, reply your action selection and communication messages in this fixed format: Action selection: Your action. Message to Team: “Your Message”. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules.\nAre you ready to begin?\""


BACKGROUND_PROMPT_CUTOFF_IMPROVED = "Welcome to our interactive text game! You are a specialist on a three-person search-and-rescue team. Your call sign is {agent_id}.\n\n\
----------------  ROLE AND OBJECTIVE  ----------------\n\
* Defuse all 5 hidden bombs as fast as possible.\n* A team earns 10 x (number of phases) points for each bomb defused.\n* Maximize the final team score.\n\n\
----------------  ENVIRONMENT MAP  ----------------  \n\
The facility is a connected graph of rooms labeled with integers:\n\
{map}\n\
\nYou may move only to rooms listed as directly connected to your current location.\n\n\
----------------  BOMB TYPES  ----------------\n\
* Bombs may have 1, 2, or 3 color phases. Defuse them by applying the matching colored tools in order.\n\n\
----------------  TOOLS PER PLAYER  ----------------\n\
* Alpha   - red, green\n* Bravo   - green, blue\n* Charlie - blue, red\n\n\
----------------  VALID ACTIONS  ----------------\n\
1. Move to an adjacent room     -> Move to Room X\n2. Inspect a bomb's phase list  -> Inspect Bomb\n3. Apply a wire cutter tool     -> Apply <Color> Tool\n\n\
----------------  COMMUNICATION  ----------------\n\
* Each round you may append ONE message to teammates (they will read it next round).\n* If communication is cut you will receive: \"Communication cutoff: no communication available this round\".\n  - Continue acting based on previous information.\n  - When communication returns, share any information missed during the cutoff.\n\n\
----------------  OBSERVATION INFO  ----------------  \n\
At the start of every round you learn:\n* Current round number and cumulative team score\n* Your room contents (bombs, players)\n* Locations of teammates\n* Messages sent in the previous round\n\n\
----------------  REPLY FORMAT  ----------------  \n\
To facilitate our interaction, reply your action selection and communication messages in this fixed format: Action selection: Your action. Message to Team: “Your Message”. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules.\nAre you ready to begin?\""


INITIAL_BELIEF_IMPROVED = "----------------  BELIEF STATE  ----------------  \n\
Role: Player {agent_id}\n\
Current round: 1   |   Team score: 0\n\
\n\
----------------  POSITION  ----------------  \n\
You are in Room {initial_node}\n\
Other players here: alpha, bravo, charlie\n\
\n\
----------------  ROOM CONTENTS  ----------------  \n\
Bomb {initial_bomb}: sequence UNKNOWN\n\
No other bombs detected in this room.\n\
Contents of all rooms:\n  \
{map_nodes}\
\n\
\n\
----------------  TEAMMATE LOCATIONS  ----------------  \n\
alpha   - Room {initial_node}\n\
bravo   - Room {initial_node}\n\
charlie - Room {initial_node}\n\
\n\
----------------  MAP (adjacency list)  ----------------  \n\
Each line shows: Room : directly connected rooms  \n\
{map}\
\n\
----------------  KNOWN BOMBS  ----------------  \n\
Bomb {initial_bomb}  |  Room {initial_node}  |  Phase list: UNKNOWN | Status: UNKNOWN\n\
(4 bombs remain unlocated, 0 bombs defused)\n\
\n\
----------------  RECENT MESSAGES  ----------------  \n\
alpha   : None\n\
bravo   : None\n\
charlie : None\n\
\n\
----------------  TOOL INVENTORY  ----------------  \n\
alpha   - red, green\n\
bravo   - green, blue\n\
charlie - red, blue\n\
\n\
----------------  INFORMATON I SHOULD SHARE  ----------------  \n\
\n\
----------------  ACTION SYNTAX  ----------------  \n\
Move         : Move to Room X\n\
Inspect Bomb : Inspect Bomb\n\
Apply tool   : Apply <Color> Tool\n\
Send message : Message to Team: \"...\"  \n\
There is no standby action, but you can inspect a bomb if there is any in the room.\
--------------------------------------------------\""


TIPS = "GENERAL COORDINATION AND MEMORY TIPS:\n\
1. Inspection Rule: If an uninspected bomb is in your current room **and two or more teammates are present**, ONLY the teammate whose call-sign comes first alphabetically performs \"Inspect Bomb\" (unless a different plan was clearly agreed). Everyone else should spend the turn on a higher-value task (move, cut, etc.).\n\
2. Memory Rule: When asked to update your belief state, add any other useful information not already contained in the previous belief state. Examples: rooms visited, bomb locations, each bomb's phase list and progress. Do not delete a bomb from your belief state, change its status to DEFUSED. Your memory of the history of events is limited, but you always keep your previous belief state.\n\
3. Bomb Counter Rule: Five bombs exist at the start. Track how many remain. Pay attention to the communication messages to keep track of bombs defused, location or progress towards defusal. Be explicit when communicating about bomb status/phases.\n\
4. No Duplicate Tools Rule: Before applying a tool, check messages and your belief state to be sure a teammate will not apply the same tool to the same bomb in the same round. Assume that the teammate whose call-sign comes first alphabetically will perform a defusal action if he is in the same room and if he has the corresponding tool.\n\
5. Sequence Coordination Rule: After every successful cut on a multi-phase bomb, broadcast the remaining color sequence. The teammate with the next required color should position in that room; others should keep exploring.\n\
6. Same Round Coordination Rule: Assume agents whose call-sign comes first alphabetically will act before you and will apply their tool if they can and use this to coordinate your tool use. Example: In a given round, agents A, B and C are in the same room with a bomb with know sequence green-red-blue. If A has the green tool, B can expect A is going to use it and try to use red if he has the red tool. The same applies to C.\n\
7. Exploration Rule: Do not cluster without reason. When no bomb needs multi-player attention, each teammate should move to a different unexplored or partially explored room to maximise information gain.\n\
8. Cutoff Recovery Rule: If you receive \"Communication cutoff\", continue acting based on memory. When comms return, send a SUMMARY listing all actions and observations since the last successful message.\n\
9. Never assume your belief state is ground truth: correct false information if directly contradicted by new evidence.\n\
10. Append to your belief state precious information you have not already shared with your team, like bomb phases/location you discovered.\n"



class ChatAgent():
    def __init__(self,agent_id='alpha',model="gpt-4-turbo-preview",temperature=0.0,message_history =None, belief = False, allow_comm = True,initial_bomb = 1, initial_node = 0, log_chat=True, log_path="data/chat_log.json", memory_size=2, cutoff=False, improved = False, tips = False, preset='default'):
        self.agent_id = agent_id
        self.model = model
        self.temperature=temperature
        self.model_supports_temperature = {}
        self.total_usage = {}
        self.log_chat=log_chat
        self.log_path = log_path
        self.memory_size = memory_size
        self.cutoff = cutoff
        self.cutoff_activated = False
        self.improved = improved
        self.tips = tips
        self.round = 1

        self.belief = belief
        # self.last_belief = INITIAL_BELIEF.format(agent_id = agent_id,initial_bomb = initial_bomb,initial_node=initial_node)
        self.allow_comm = allow_comm
        if self.allow_comm:
            if self.cutoff:
                if self.improved:
                    BACKGROUND_PROMPT = BACKGROUND_PROMPT_CUTOFF_IMPROVED
                else:
                    BACKGROUND_PROMPT = BACKGROUND_PROMPT_CUTOFF
            else:
                if self.improved:
                    BACKGROUND_PROMPT = BACKGROUND_PROMPT_NEW_IMPROVED
                else:
                    BACKGROUND_PROMPT = BACKGROUND_PROMPT_NEW
            if self.tips:
                BACKGROUND_PROMPT += '\n\n' + TIPS
            if self.improved:
                self.last_belief = INITIAL_BELIEF_IMPROVED.format(agent_id=agent_id, initial_bomb=initial_bomb,initial_node=initial_node,map=PRESET_MAPS[preset][0],map_nodes=PRESET_MAPS[preset][1])
            else:
                self.last_belief = INITIAL_BELIEF.format(agent_id = agent_id,initial_bomb = initial_bomb,initial_node=initial_node)
        else:
            BACKGROUND_PROMPT = BACKGROUND_PROMPT_NOCOMM
            self.last_belief = INITIAL_BELIEF_NOCOMM.format(agent_id=agent_id, initial_bomb=initial_bomb,
                                                    initial_node=initial_node)
        if message_history is None:
            self.message_history = [
                {"role": "system", "content": 'You are playing a text game with the user.'},
                {"role": "user", "content": BACKGROUND_PROMPT.format(agent_id = agent_id, map=PRESET_MAPS[preset][0])},
                # {"role": "user", "content": INSTRUCT_PROMPT},
            ]
            if self.belief:
                self.message_history.append({"role": "user", "content": self.last_belief+INITIAL_PROMPT.format(agent_id = self.agent_id)})
        else:
            self.message_history = message_history

    @staticmethod
    def _usage_to_dict(usage_obj) -> dict:
        """
        Convert the `CompletionUsage` object into a flat dict that contains
        only entries whose numeric value > 0.

        Nested `*_tokens_details` blocks are flattened with a dotted name:
            prompt_tokens_details.cached_tokens  → 0   (filtered out)
            completion_tokens_details.audio_tokens → 7 → kept as
                'completion_tokens_details.audio_tokens': 7
        """
        # 1) turn the Pydantic model (or any object) into a python dict
        if hasattr(usage_obj, "model_dump"):       # pydantic v2 (OpenAI ≥ 1.0.0)
            raw = usage_obj.model_dump(exclude_none=True)
        elif hasattr(usage_obj, "dict"):           # pydantic v1 fallback
            raw = usage_obj.dict(exclude_none=True)
        else:                                      # plain object → __dict__
            raw = vars(usage_obj)

        # 2) recursively flatten & filter
        def flatten(d: Mapping, prefix: str = ""):
            for k, v in d.items():
                name = f"{prefix}.{k}" if prefix else k
                if isinstance(v, Mapping):
                    yield from flatten(v, name)
                elif isinstance(v, Number) and v > 0:
                    yield name, v

        return dict(flatten(raw))
    
    def _bump_usage(self, usage_obj) -> None:
        """Accumulate the latest call’s `usage` into `self.total_usage`."""
        for key, value in self._usage_to_dict(usage_obj).items():
                self.total_usage[key] = self.total_usage.get(key, 0) + value

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def makeAPIcall(self):

        supports_temp = self.model_supports_temperature.get(self.model, True)

        try:
            if supports_temp:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                    temperature=self.temperature,
                )
            else:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                )
        except openai.BadRequestError as e:
            if "Unsupported" in str(e):
                self.model_supports_temperature[self.model] = False
                # Retry without temperature
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                )
            else:
                raise
        
        if self.log_chat:    
            # Append prompt and response to a JSON file for logging
            log_entry = {
                'agent': self.agent_id,
                'round': self.round,
                'prompt': self.message_history if self.message_history else "",
                'response': response.choices[0].message.content
            }
            try:
                if os.path.exists(self.log_path):
                    with open(self.log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")
                else:
                    with open(self.log_path, "w", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Logging failed: {e}")
            try:
                self._bump_usage(response.usage)
            except Exception as e:
                print(f"Usage bumping failed: {e}")
        return response.choices[0].message.content

    def update_history(self,text):

        if self.belief:
            text += UPDATE_PROMPT
            if self.model == 'gpt-3.5-turbo':
                last_action = self.message_history.pop()['content']
                self.message_history.append({"role": "user", "content": last_action+ text})
                new_belief = self.makeAPIcall()
                self.last_belief = new_belief
                self.message_history.pop()
                self.message_history.append({"role": "assistant", "content": last_action})
                self.message_history.append({"role": "user", "content": 'Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. '+new_belief+INITIAL_PROMPT.format(agent_id = self.agent_id)})
            else:
                self.message_history.append({"role": "user", "content": text})
                new_belief = self.makeAPIcall()
                self.last_belief = new_belief
                self.message_history.pop()
                content = new_belief
                if self.allow_comm and self.cutoff_activated:
                    content += ' ' + CUTOFF
                self.message_history.append({"role": "user", "content": content+' '+INITIAL_PROMPT.format(agent_id = self.agent_id)})

            # new_belief = self.update_belief(text)
            # self.message_history.append({"role": "user", "content": new_belief+INITIAL_PROMPT})

        else:
            if self.allow_comm and self.cutoff_activated:
                text += ' ' + CUTOFF
            text += "Given the above observation, what is your next action?"
            print('env', text)
            self.message_history.append({"role": "user", "content": text})
            new_belief = "None"


        # delete oldest round in memory
        if self.memory_size != -1 and len(self.message_history)>2+self.memory_size*2+2: # message_history: index 0 and 1 are game rules + 2 items per round (belief and action) + 2 items of the current round (belief and action) + 1 observation after action
            self.message_history.pop(2)
            self.message_history.pop(2)
        return new_belief

    def step(self):
        action = self.makeAPIcall()

        self.message_history.append({"role": "assistant", "content": action})
        print(self.agent_id,action)
        self.round += 1
        return action

    def save(self,data_path):
        data = {}
        data['agent_id'] = self.agent_id
        data['model'] = self.model
        data['temperature'] = self.temperature
        data['message_history'] = self.message_history
        data['belief'] = self.belief
        data['allow_comm'] = self.allow_comm
        timestr = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        path = data_path + '{model}_{temperature}_{allow_comm}_{agent_id}_{timestr}.json'.format(timestr=timestr,agent_id = self.agent_id,allow_comm = self.allow_comm,model = self.model,temperature = self.temperature)
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump(data, f)

    def ask(self, question_text):
        self.message_history.append({"role": "user", "content": question_text})
        reply = self.makeAPIcall()
        self.message_history.pop()
        return reply


    def read_belief(self):
        return self.message_history
