import argparse
import json
import os
from dragonTextEnv import DragonTextEnv, ChatAgent
import numpy as np
import time
from pathlib import Path
import utils


parser = argparse.ArgumentParser(description='Text interface for LLM agent')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--save_path', default='data/', type=str,
                    help='path of saved log files')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for environment initialization')
parser.add_argument('--tom', action='store_true', default=False,
                    help='conduct belief inference measurements')
parser.add_argument('--belief', action='store_true', default=False,
                    help='maintain a belief state about the env')
parser.add_argument('--allow_comm', action='store_true', default=False,
                    help='allow communication between team members')
parser.add_argument('--model', type=str, default='gpt-4-turbo-preview',
                    help='base LM to use')
parser.add_argument('--include_agent_action', action='store_true', default=False,
                    help='present other agents action in obs')
# parser.add_argument('--act_and_comm', action='store_true', default=False,
#                     help='allow action and communication in the same round')
parser.add_argument('--tool_per_agent', type = int, default=2,
                    help='kinds of tool each agent has')
parser.add_argument('--temperature', type = float, default=0,
                    help='temperature used by base LM')
parser.add_argument('--max_step', type = int, default=30,
                    help='maximum steps allowed')
parser.add_argument('--exp_name', type = str, default='default_exp',
                    help='exp name to save')
parser.add_argument('--cutoff', type = float, default=0.0,
                    help='probability of communication cutoff every round, e.g., 0.5 means commnunication in not available 50% of the time.')
parser.add_argument('--memory_size', type = int, default=2,
                    help='how many previous rounds to keep in memory for each agent, e.g., 2 means the agent can access the last 2 rounds of history. -1 means no limit.')
parser.add_argument('--tom_reasoning', action='store_true', default=False,
                    help='enable theory of mind reasoning for agents, i.e., agents are asked questions about other agents\' knowledge and belief before communicatig')

# model
args = parser.parse_args()

exp_base   = args.exp_name
save_root  = Path(args.save_path)          # Path object once, reuse it
args.exp_name = utils.next_experiment_name(exp_base, save_root, args.model, args.seed)

print(args)

DATA_PATH = os.path.join(args.save_path ,args.model ,args.exp_name,'seed' + str(args.seed))+'/'
# DATA_PATH += 'GPT4-turbo-comm-seed24/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

chat_log_path = os.path.join(DATA_PATH, 'chat_log.jsonl')
    
with open(os.path.join(DATA_PATH, 'args.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, indent=2)
    
renders_path = os.path.join(DATA_PATH, 'renders')
if not os.path.exists(renders_path):
    os.makedirs(renders_path)

seed = args.seed
ToM = args.tom
belief = args.belief
allow_comm = args.allow_comm
model_name = args.model
act_and_comm = True
memory_size = args.memory_size
cutoff = args.cutoff
tom_reasoning = args.tom_reasoning

np.random.seed(seed)

env = DragonTextEnv(seed = seed,include_agent_action = args.include_agent_action,allow_comm = allow_comm,act_and_comm = act_and_comm,tool_per_agent = args.tool_per_agent)
Action = env.env.action_enum
obs = env.env._get_obs()
info = {}
initial_node = str(env.env.agents['alpha'].node.id)
initial_bomb = str(env.env.agents['alpha'].bomb.id)
chat_agents = {'alpha':ChatAgent(agent_id='alpha',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm, initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15),
               'bravo':ChatAgent(agent_id='bravo',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15),
               'charlie':ChatAgent(agent_id='charlie',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15)}
initial_actions = {'alpha':Action.go_to(int(initial_node)),'bravo':Action.go_to(int(initial_node)),'charlie':Action.go_to(int(initial_node))}
communications = {'alpha':'None','bravo':'None','charlie':'None'}

chat_output = {}
actions = {}
done = {'__all__':False}
round = 1
invalid_actions = 0
total_actions = 0
tom_questions = None

cols = ['round', 'agent_id',
 'chat_output', 'action', 'comm','obs_text','new_belief','ToM1st','ToM2nd','ToM3rd', 'ToM1st_q', 'ToM2nd_q', 'ToM3rd_q', 'ground_truth']


with open(DATA_PATH + 'summary.csv', 'w+', encoding='utf-8') as f:
    for k in cols:
        f.write(k)
        f.write(',')
    f.write('\n')



who_has_inspected_what = {'alpha':set(),'bravo':set(),'charlie':set()}

while not done['__all__'] and round <= args.max_step:
    print(f"{60*'-'}Start of Round {round}:{60*'-'}")
    env.env.render(overlay_graph=True, save_path=os.path.join(renders_path, f'round_{round}.pdf'))
    #input(f"Press Enter to continue")
    cutoff_activated = np.random.random() < cutoff
    for agent_id in chat_agents.keys():
        chat_agent = chat_agents[agent_id]
        chat_agent.cutoff_activated = cutoff_activated
        env.cutoff_activated = chat_agent.cutoff_activated
        if round == 1 and not belief:
            _, reward, done, info, obs_text, valid_action = env.step(agent_id, 0,initial_actions, communications)

            chat_agent.update_history(obs_text)

        # if agent_id not in initial_actions.keys():
        chat_output[agent_id] = chat_agent.step()

        initial_actions[agent_id], communications[agent_id] = env.decode_action(chat_output[agent_id])
        # initial_actions[agent_id], communications[agent_id] = env.decode_action_API(chat_output[agent_id])

        if tom_reasoning and tom_questions is not None and not env.cutoff_activated:
            _, reward, done, info, obs_text, valid_action = env.step(agent_id, round,initial_actions, communications, tom_questions)
        else:
            _, reward, done, info, obs_text, valid_action = env.step(agent_id, round,initial_actions, communications)

        if not valid_action:
            print(f"Invalid action for agent {agent_id} in round {round}.")
            invalid_actions += 1
            
        total_actions += 1

        new_belief = chat_agent.update_history(obs_text)

        agent = env.env.agents[agent_id]



        ground_truth = None
        ToM1st = None
        ToM2nd = None
        ToM3rd = None
        ToM1st_q = None
        ToM2nd_q = None
        ToM3rd_q = None

        if ToM:
            target_id = np.random.choice([x for x in chat_agents.keys() if x != agent_id])
            if initial_actions[agent_id] is None:
                ground_truth = None
                ToM1st = None
                ToM2nd = None
                ToM3rd = None
                ToM1st_q = None
                ToM2nd_q = None
                ToM3rd_q = None

            elif initial_actions[agent_id].node() is not None:
                if agent.node.id != initial_actions[agent_id].node().id:
                    ground_truth = False
                else:
                    ground_truth = True

                # first-order ToM / introspective
                ToM1st_q = 'Do you know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id)
                ToM1st = chat_agent.ask(obs_text + ToM1st_q)
                # second-order ToM
                ToM2nd_q = 'Does player {player_id} know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id)
                ToM2nd = chat_agent.ask(obs_text + ToM2nd_q)
                # third-order ToM
                ToM3rd_q = 'Based on the observation and previous history, is player {player_id} aware of the fact that you know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id)
                ToM3rd = chat_agent.ask(obs_text + ToM3rd_q)
            elif initial_actions[agent_id].tool() is not None:
                ground_truth = False
                if agent.bomb:
                    bomb_id = agent.bomb.id

                    # first-order ToM / introspective
                    ToM1st_q = 'Do you know the state and remaining sequence of bomb {bomb_id} has been changed?'.format(
                            player_id=target_id, bomb_id=bomb_id)
                    ToM1st = chat_agent.ask(obs_text + ToM1st_q)

                    # second-order ToM
                    ToM2nd_q = 'Does player {player_id} know the state and remaining sequence of bomb {bomb_id} has been changed?'.format(
                            player_id=target_id, bomb_id=bomb_id)
                    ToM2nd = chat_agent.ask(obs_text + ToM2nd_q)
                    # third-order ToM
                    ToM3rd_q = 'Based on the observation and previous history, is player {player_id} aware of the fact that you have changed the state and remaining sequence of bomb {bomb_id}?'.format(
                            player_id=target_id,
                            bomb_id=bomb_id)
                    ToM3rd = chat_agent.ask(obs_text + ToM3rd_q)
                elif isinstance(reward, int):
                    if reward >= 0:
                        ToM1st_q = 'Do you know a bomb phase has just been defused?'
                        ToM1st = chat_agent.ask(obs_text + ToM1st_q)

                        # second-order ToM
                        ToM2nd_q = 'Does player {player_id} know a bomb phase has just been defused?'.format(player_id = target_id)
                        ToM2nd = chat_agent.ask(obs_text + ToM2nd_q)
                        # third-order ToM
                        ToM3rd_q = 'Based on the observation and previous history, is player {player_id} aware of the fact that you know a bomb phase has just been defused?'.format(player_id = target_id)
                        ToM3rd = chat_agent.ask(obs_text + ToM3rd_q)


            elif initial_actions[agent_id] == Action.inspect_bomb:
                if agent.bomb:
                    bomb_id = agent.bomb.id
                    who_has_inspected_what[agent_id].add(bomb_id)
                    ground_truth = bomb_id in who_has_inspected_what[target_id]
                    # first-order ToM / introspective
                    ToM1st_q = 'Do you know the sequence of bomb {bomb_id}?'.format(bomb_id=bomb_id)
                    ToM1st = chat_agent.ask(obs_text + ToM1st_q)
                    # second-order ToM
                    ToM2nd_q = 'Does player {player_id} know the sequence of bomb {bomb_id}?'.format(player_id=target_id,
                                                                                              bomb_id=bomb_id)
                    ToM2nd = chat_agent.ask(obs_text + ToM2nd_q)
                    # third-order ToM
                    ToM3rd_q = 'Based on the observation and previous history, is player {player_id} aware of the fact that you know the sequence of bomb {bomb_id}?'.format(player_id=target_id,
                                                                                              bomb_id=bomb_id)
                    ToM3rd = chat_agent.ask(obs_text + ToM3rd_q)
                else:
                    ground_truth = None
                    ToM1st = None
                    ToM2nd = None
                    ToM3rd = None
                    ToM1st_q = None
                    ToM2nd_q = None
                    ToM3rd_q = None
            else:
                ground_truth =None
                ToM1st = None
                ToM2nd = None
                ToM3rd = None
                ToM1st_q = None
                ToM2nd_q = None
                ToM3rd_q = None
            
            tom_questions = [ToM1st_q, ToM2nd_q, ToM3rd_q]






        # chat_agent.save(DATA_PATH)

        if initial_actions[agent_id] is None:
            class InvalidAction:
                name = "Invalid"
            initial_actions[agent_id] = InvalidAction()

        record = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':initial_actions[agent_id].name.replace('_', ' '),'comm':communications[agent_id],'obs_text': obs_text,"new_belief":new_belief,'ToM1st':ToM1st, 'ToM2nd':ToM2nd, 'ToM3rd':ToM3rd, 'ToM1st_q': ToM1st_q, 'ToM2nd_q': ToM2nd_q, 'ToM3rd_q': ToM3rd_q, 'ground_truth': ground_truth}
        with open(DATA_PATH + 'record.jsonl', 'a+', encoding='utf-8') as f:
            json.dump(record, f)
            f.write('\n')

        summary = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':initial_actions[agent_id].name.replace('_', ' '),'comm':communications[agent_id],'obs_text': obs_text,"new_belief":new_belief,'ToM1st':ToM1st, 'ToM2nd':ToM2nd, 'ToM3rd':ToM3rd, 'ToM1st_q': ToM1st_q, 'ToM2nd_q': ToM2nd_q, 'ToM3rd_q': ToM3rd_q, 'ground_truth': ground_truth}
        print(summary)
        with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
            for k,v in summary.items():
                f.write(str(v).replace(',',';').replace('\n',''))
                f.write(',')
            f.write('\n')

    round +=1

# Compute token usage
total_usage = {}
for agent_id, chat_agent in chat_agents.items():
    usage = chat_agent.total_usage
    for key, value in usage.items():
        total_usage[key] = total_usage.get(key, 0) + (value or 0)

results = {'score': env.env.score, 'rounds': round - 1, 'invalid_actions': invalid_actions, 'total_actions': total_actions, 'valid_action_rate': (total_actions - invalid_actions) / total_actions if total_actions > 0 else 0}
results.update(total_usage)

with open(os.path.join(DATA_PATH, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)