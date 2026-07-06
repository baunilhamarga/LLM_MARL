import argparse
import json
import os
from dragonTextEnv import DragonTextEnv, ChatAgent, PRESETS
import numpy as np
import time
import platform
import resource
import socket
import sys
from pathlib import Path
import utils.utils as utils
from utils.model_provider import ChatModelClient, aggregate_metrics


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
parser.add_argument('--model', type=str, default=None,
                    help='base LM to use (provider-specific default when omitted)')
parser.add_argument('--provider', choices=['openai', 'groq', 'local', 'openai-compatible'], default='openai',
                    help='inference provider (default: openai)')
parser.add_argument('--base_url', type=str, default=None,
                    help='base URL for an OpenAI-compatible inference server')
parser.add_argument('--api_key_env', type=str, default=None,
                    help='environment variable containing the provider API key')
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='Hugging Face model ID or local snapshot path for local inference')
parser.add_argument('--model_cache_dir', type=str, default='/usr/users/xai/gama_hei/projects/llm_dce/models/huggingface',
                    help='Hugging Face cache containing local model weights')
parser.add_argument('--local_dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='float16',
                    help='weight dtype for local inference (default: float16)')
parser.add_argument('--max_completion_tokens', type=int, default=None,
                    help='maximum generated tokens per model call (local default: 256)')
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
                    help='probability of communication cutoff every round, e.g., 0.5 means communication is unavailable half of the time.')
parser.add_argument('--memory_size', type = int, default=2,
                    help='how many previous rounds to keep in memory for each agent, e.g., 2 means the agent can access the last 2 rounds of history. -1 means no limit.')
parser.add_argument('--tom_reasoning', action='store_true', default=False,
                    help='enable theory of mind reasoning for agents, i.e., agents are asked questions about other agents\' knowledge and belief before communicatig')
parser.add_argument('--improved', action='store_true', default=False,
                    help='enable improved agent behavior')
parser.add_argument('--tips', action='store_true', default=False,
                    help='enable tips for agents')
parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()), default='default',
                    help='choose the preset map size')
parser.add_argument('--graph_compression', action='store_true', default=False,
                    help='enable graph compression for the environment, which allows agents to have a compact view of the environment')
parser.add_argument('--k', type=int, default=2,
                    help='number of regions (supernodes) in the graph compression if activated')
parser.add_argument('--compression_method', type=str, default='balanced_bfs',
                    help='method for graph compression (e.g., kernighan-lin, louvain, spectral_k, etc.)')
parser.add_argument('--include_visited_nodes', action='store_true', default=False,
                    help='include information about per-agent visited nodes in the agent observations')

# model
args = parser.parse_args()
if args.model is None:
    args.model = 'Llama-3.1-8B-Instruct' if args.provider == 'local' else 'gpt-4-turbo-preview'

exp_base   = args.exp_name
save_root  = Path(args.save_path)          # Path object once, reuse it
args.exp_name = utils.next_experiment_name(exp_base, save_root, args.model, args.seed)
exp_name = args.exp_name

print(args)

experiment_wall_started = time.perf_counter()
experiment_cpu_started = time.process_time()
experiment_resource_started = resource.getrusage(resource.RUSAGE_SELF)

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
provider = args.provider
model_client = ChatModelClient(
    provider=provider,
    base_url=args.base_url,
    api_key_env=args.api_key_env,
    model_path=args.model_path if provider == 'local' else None,
    model_cache_dir=args.model_cache_dir if provider == 'local' else None,
    local_dtype=args.local_dtype,
    max_completion_tokens=args.max_completion_tokens,
)
# Validate credentials and endpoint configuration before the experiment loop.
model_client.prepare(model_name)
act_and_comm = True
memory_size = args.memory_size
cutoff = args.cutoff
tom_reasoning = args.tom_reasoning
improved = args.improved
tips = args.tips
preset = args.preset
graph_compression = args.graph_compression
compression_method = args.compression_method
k = args.k
include_visited_nodes = args.include_visited_nodes

np.random.seed(seed)

env = DragonTextEnv(seed = seed,include_agent_action = args.include_agent_action,allow_comm = allow_comm,act_and_comm = act_and_comm,tool_per_agent = args.tool_per_agent, preset = preset, graph_compression = graph_compression, k = k, compression_method = compression_method, include_visited_nodes=include_visited_nodes)
Action = env.env.action_enum
obs = env.env._get_obs()
info = {}
initial_node = str(env.env.agents['alpha'].node.id)
initial_bomb = str(env.env.agents['alpha'].bomb.id)
if graph_compression:
    initial_region = env.compressed_graph.region_of(int(initial_node))
    initial_view = env.compressed_graph.region_adjlist_str(initial_region)
    nodes_per_region_str = env.compressed_graph.nodes_per_region_str()
else:
    initial_region = None
    initial_view = None
    nodes_per_region_str = None
chat_agents = {'alpha':ChatAgent(agent_id='alpha',model = model_name,provider=provider,base_url=args.base_url,api_key_env=args.api_key_env,model_client=model_client,temperature=args.temperature,belief = belief,allow_comm = allow_comm, initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15, improved=improved, tips=tips, preset=preset, graph_compression=graph_compression, initial_view=initial_view, initial_region=initial_region, nodes_per_region_str=nodes_per_region_str),
               'bravo':ChatAgent(agent_id='bravo',model = model_name,provider=provider,base_url=args.base_url,api_key_env=args.api_key_env,model_client=model_client,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15, improved=improved, tips=tips, preset=preset, graph_compression=graph_compression, initial_view=initial_view, initial_region=initial_region, nodes_per_region_str=nodes_per_region_str),
               'charlie':ChatAgent(agent_id='charlie',model = model_name,provider=provider,base_url=args.base_url,api_key_env=args.api_key_env,model_client=model_client,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node,log_path = chat_log_path, memory_size = memory_size, cutoff=cutoff>1e-15, improved=improved, tips=tips, preset=preset, graph_compression=graph_compression, initial_view=initial_view, initial_region=initial_region, nodes_per_region_str=nodes_per_region_str)}
initial_actions = {'alpha':Action.go_to(int(initial_node)),'bravo':Action.go_to(int(initial_node)),'charlie':Action.go_to(int(initial_node))}
communications = {'alpha':'None','bravo':'None','charlie':'None'}
for agent_id in chat_agents.keys():
    env.visited_nodes[agent_id] = [int(initial_node)]

chat_output = {}
actions = {}
done = {'__all__':False}
round = 1
invalid_actions = 0
total_actions = 0
tom_questions = None
tom_answers = None
tom_agents = {}
results = {}

cols = ['round', 'agent_id',
 'chat_output', 'action', 'comm','obs_text','new_belief','ToM1st','ToM2nd','ToM3rd', 'ToM1st_q', 'ToM2nd_q', 'ToM3rd_q', 'ground_truth']


with open(DATA_PATH + 'summary.csv', 'w+', encoding='utf-8') as f:
    for k in cols:
        f.write(k)
        f.write(',')
    f.write('\n')

if graph_compression:
    env.compressed_graph.draw(path = os.path.join(DATA_PATH, 'compressed_graph.pdf'))

who_has_inspected_what = {'alpha':set(),'bravo':set(),'charlie':set()}

while not done['__all__'] and round <= args.max_step:
    print(f"{60*'-'}Start of Round {round}:{60*'-'}")
    env.env.render(overlay_graph=True, save_path=os.path.join(renders_path, f'round_{round}.pdf'))
    for agent_id in chat_agents.keys():
        chat_agent = chat_agents[agent_id]
        if round == 1 and not belief:
            _, reward, done, info, obs_text, valid_action, _ = env.step(agent_id, 0,initial_actions, communications)

            chat_agent.update_history(obs_text)

        # if agent_id not in initial_actions.keys():
        chat_output[agent_id] = chat_agent.step()

        initial_actions[agent_id], communications[agent_id] = env.decode_action(chat_output[agent_id], agent_id=agent_id)
        # initial_actions[agent_id], communications[agent_id] = env.decode_action_API(chat_output[agent_id])

        _, reward, done, info, obs_text, valid_action, results[agent_id] = env.step(agent_id, round,initial_actions, communications, tom_questions=tom_questions, tom_reasoning=tom_reasoning)

        if not valid_action:
            print(f"Invalid action for agent {agent_id} in round {round}.")
            invalid_actions += 1
            
        total_actions += 1


        agent = env.env.agents[agent_id]



        ground_truth = None
        ToM1st = None
        ToM2nd = None
        ToM3rd = None
        ToM1st_q = None
        ToM2nd_q = None
        ToM3rd_q = None

        chat_agent.round = round

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
            tom_answers = [ToM1st, ToM2nd, ToM3rd]
            tom_agents[agent_id] = (tom_questions, tom_answers)

    cutoff_activated = np.random.random() < cutoff
    env.cutoff_activated = cutoff_activated 
    # Update beliefs after all agents have acted       
    for agent_id in chat_agents.keys():
        chat_agent.cutoff_activated = env.cutoff_activated
        chat_agent = chat_agents[agent_id]
        obs_text = env.step_text(agent_id, round, initial_actions, communications, results[agent_id])
        last_belief = chat_agent.last_belief
        new_belief = chat_agent.update_history(obs_text)
        chat_agent.round += 1
        
        if ToM:
            ToM1st_q, ToM2nd_q, ToM3rd_q = tom_agents[agent_id][0]
            ToM1st, ToM2nd, ToM3rd = tom_agents[agent_id][1]
        else:
            ToM1st_q = ToM2nd_q = ToM3rd_q = None
            ToM1st = ToM2nd = ToM3rd = None

        record = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':initial_actions[agent_id].name.replace('_', ' ') if initial_actions[agent_id] != None else 'Invalid','comm':communications[agent_id],'obs_text': obs_text,"new_belief":last_belief,'ToM1st':ToM1st, 'ToM2nd':ToM2nd, 'ToM3rd':ToM3rd, 'ToM1st_q': ToM1st_q, 'ToM2nd_q': ToM2nd_q, 'ToM3rd_q': ToM3rd_q, 'ground_truth': ground_truth}
        with open(DATA_PATH + 'record.jsonl', 'a+', encoding='utf-8') as f:
            json.dump(record, f)
            f.write('\n')

        summary = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':initial_actions[agent_id].name.replace('_', ' ') if initial_actions[agent_id] != None else 'Invalid','comm':communications[agent_id],'obs_text': obs_text,"new_belief":last_belief,'ToM1st':ToM1st, 'ToM2nd':ToM2nd, 'ToM3rd':ToM3rd, 'ToM1st_q': ToM1st_q, 'ToM2nd_q': ToM2nd_q, 'ToM3rd_q': ToM3rd_q, 'ground_truth': ground_truth}
        print(summary)
        with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
            for k,v in summary.items():
                f.write(str(v).replace(',',';').replace('\n',''))
                f.write(',')
            f.write('\n')

    round +=1

# Aggregate token usage and provider-independent request metrics.
total_usage = {}
agent_model_metrics = {}
aggregate_model_metrics = {
    'provider': provider,
    'model': model_name,
    'request_count': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'responses_with_usage': 0,
    'responses_without_usage': 0,
    'usage_collection_errors': 0,
    'wall_time_seconds': 0.0,
}
for agent_id, chat_agent in chat_agents.items():
    agent_model_metrics[agent_id] = chat_agent.metrics_summary()
    usage = chat_agent.total_usage
    aggregate_metrics(total_usage, usage)
    for key in aggregate_model_metrics:
        if key in ('provider', 'model'):
            continue
        aggregate_model_metrics[key] += chat_agent.model_metrics.get(key, 0)

aggregate_model_metrics['usage'] = total_usage
aggregate_model_metrics['agents'] = agent_model_metrics
aggregate_model_metrics['setup'] = model_client.setup_metrics
if total_usage.get('generation_time_seconds', 0) > 0:
    aggregate_model_metrics['completion_tokens_per_second'] = (
        total_usage.get('completion_tokens', 0) / total_usage['generation_time_seconds']
    )

process_usage = resource.getrusage(resource.RUSAGE_SELF)
runtime_metrics = {
    'wall_time_seconds': time.perf_counter() - experiment_wall_started,
    'process_cpu_time_seconds': time.process_time() - experiment_cpu_started,
    'user_cpu_time_seconds': process_usage.ru_utime - experiment_resource_started.ru_utime,
    'system_cpu_time_seconds': process_usage.ru_stime - experiment_resource_started.ru_stime,
    'peak_rss_mb': process_usage.ru_maxrss / 1024,
    'hostname': socket.gethostname(),
    'logical_cpu_count': os.cpu_count(),
    'platform': platform.platform(),
    'python_version': sys.version.split()[0],
}

results = {'score': env.env.score, 'rounds': round - 1, 'invalid_actions': invalid_actions, 'total_actions': total_actions, 'valid_action_rate': (total_actions - invalid_actions) / total_actions if total_actions > 0 else 0}
results.update(total_usage)
results['model_metrics'] = aggregate_model_metrics
results['runtime_metrics'] = runtime_metrics

with open(os.path.join(DATA_PATH, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
    
print(f"Experiment {exp_name} completed. Results saved to {DATA_PATH}.")
