from gym_dragon.core import Region
from gym_dragon.envs import DragonEnv, MiniDragonEnv
from gym_dragon.wrappers import MiniObs
import os
import numpy as np
import argparse
from tqdm import trange
import pandas as pd

PRESETS = {
    'village': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63), 11: (32, 56), 12: (41, 52), 14: (15, 67), 15: (21, 55), 16: (45, 52), 17: (5, 62), 20: (42, 60), 21: (6, 56), 22: (48, 52), 23: (37, 78), 24: (40, 74), 26: (46, 65), 27: (48, 60), 28: (11, 69), 31: (14, 80), 32: (46, 75), 33: (6, 69), 34: (20, 80), 36: (48, 70), 38: (21, 69), 40: (6, 76), 41: (23, 76), 43: (46, 80), 45: (28, 88), 46: (43, 91), 47: (29, 76), 48: (15, 92), 49: (9, 87), 51: (47, 85), 52: (4, 82), 53: (32, 69), 55: (34, 92), 56: (9, 82), 57: (31, 82), 58: (39, 89), 63: (46, 90), 64: (48, 90), 65: (27, 96), 66: (20, 96), 67: (44, 98), 68: (5, 95), 69: (11, 98), 70: (39, 98), 71: (34, 98), 72: (48, 98), 73: (34, 75), 75: (6, 98), 77: (6, 90)},
    'default': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63)},
    'easy': {0: (28, 56), 8: (22, 63), 6: (33, 64)},
    'medium': {23: (37, 78), 34: (20, 80), 38: (21, 69), 41: (23, 76), 47: (29, 76), 53: (32, 69), 57: (31, 82), 73: (34, 75)},
    'hard': {0: (28, 56), 3: (15, 55), 5: (40, 56), 6: (33, 64), 8: (22, 63), 14: (15, 67), 20: (42, 60), 23: (37, 78), 31: (14, 80), 34: (20, 80), 38: (21, 69), 41: (23, 76), 47: (29, 76), 53: (32, 69), 57: (31, 82), 73: (34, 75)},
}

def random_baseline(seed=0, num_episodes=1, max_rounds=30, render=False, performances=pd.DataFrame(columns=['seed', 'episode', 'score', 'rounds']), preset="default"):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    if render: 
        renders_path = './tmp/renders'
        os.makedirs(renders_path, exist_ok=True)

    env = MiniDragonEnv(mission_length=max_rounds+1,
                        recon_phase_length=0,
                        include_chained_bombs=False,
                        seconds_per_timestep=1.0,
                        include_fire_bombs=False,
                        include_fuse_bombs=False,
                        color_tools_only=True,
                        obs_wrapper=MiniObs,
                        valid_nodes=PRESETS[preset])
    env.seed(seed)
    for episode in range(num_episodes):
        obs = env.reset()
        done = {agent_id: False for agent_id in env.get_agent_ids()}
        with trange(1, max_rounds + 1, desc=f"Episode {episode+1}/{num_episodes}", unit="round") as t:
            for round in t:
                if render:
                    renders_dir = os.path.join(renders_path, f'seed_{seed}', f'episode_{episode+1}')
                    os.makedirs(renders_dir, exist_ok=True)
                    env.render(overlay_graph=True, save_path=os.path.join(renders_dir, f'round_{round}.pdf'))
                random_action = env.action_space_sample()
                obs, reward, done, info = env.step(random_action)
                if all(done.values()):
                    t.set_postfix_str(f"Score: {env.score}")
                    break
        print('Episode:', episode+1, 'Score:', env.score, 'Total Rounds:', round)
        performances.loc[len(performances)] = {'seed': seed, 'episode': episode+1, 'score': env.score, 'rounds': round}

    if num_episodes > 0:
        filtered_performances = performances[performances['seed'] == seed]
        avg_score = np.mean(filtered_performances['score'])
        std_score = np.std(filtered_performances['score'])
        print(f'Average Score: {avg_score:.2f}, Std_dev: {std_score:.2f}')

        avg_rounds = np.mean(filtered_performances['rounds'])
        std_rounds = np.std(filtered_performances['rounds'])
        print(f'Average Rounds: {avg_rounds:.2f}, Std_dev: {std_rounds:.2f}')

    return performances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Initial random seed')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes per seed')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds to run')
    parser.add_argument('--max_rounds', type=int, default=999, help='Maximum rounds per episode')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()), default='default')
    parser.add_argument('--render', action='store_true', help='Render and save environment visuals')
    args = parser.parse_args()
    initial_seed = args.seed
    num_episodes = args.num_episodes
    max_rounds = args.max_rounds
    num_seeds = args.num_seeds
    render = args.render
    preset = args.preset

    performances = pd.DataFrame(columns=['seed', 'episode', 'score', 'rounds'])
    
    for seed in range(initial_seed, initial_seed + num_seeds):
        print(f'Progress: {seed - initial_seed + 1}/{num_seeds}')
        print(f'Running random baseline for {num_episodes} episodes with seed: {seed}')
        np.random.seed(seed)
        random_baseline(seed=seed, num_episodes=num_episodes, max_rounds=max_rounds, render=render, performances=performances, preset=preset)

    if not performances.empty:
        avg_score = performances['score'].mean()
        std_score = performances['score'].std()
        avg_rounds = performances['rounds'].mean()
        std_rounds = performances['rounds'].std()
        print(f'\nExperiment Results:')
        print(f'Average Score: {avg_score:.2f} ± {std_score:.2f}')
        print(f'Average Rounds: {avg_rounds:.2f} ± {std_rounds:.2f}')
        # Compute per-seed averages
        per_seed_avg = performances.groupby('seed').agg({'score': 'mean', 'rounds': 'mean'})
        avg_of_avgs_score = per_seed_avg['score'].mean()
        std_of_avgs_score = per_seed_avg['score'].std()
        avg_of_avgs_rounds = per_seed_avg['rounds'].mean()
        std_of_avgs_rounds = per_seed_avg['rounds'].std()
        print(f'Average of per-seed average scores: {avg_of_avgs_score:.2f} ± {std_of_avgs_score:.2f}')
        print(f'Average of per-seed average rounds: {avg_of_avgs_rounds:.2f} ± {std_of_avgs_rounds:.2f}')
    else:
        print('No performances recorded.')