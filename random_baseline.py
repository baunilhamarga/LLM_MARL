from gym_dragon.core import Region
from gym_dragon.envs import DragonEnv, MiniDragonEnv
from gym_dragon.wrappers import MiniObs
import os
import numpy as np

def random_baseline(seed=0, num_episodes=1, max_rounds=30):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    renders_path = './tmp/renders'
    os.makedirs(renders_path, exist_ok=True)
    env = MiniDragonEnv(mission_length=999,
                        recon_phase_length=0,
                        include_chained_bombs=False,
                        include_fire_bombs=False,
                        include_fuse_bombs=False,
                        color_tools_only=True,
                        obs_wrapper=MiniObs)
    env.seed(seed)

    for episode in range(num_episodes):
        round = 1
        obs = env.reset()
        done = {agent_id: False for agent_id in env.get_agent_ids()}
        while not all(done.values()) and round <= max_rounds:
            env.render(overlay_graph=True, save_path=os.path.join(renders_path, f'episode_{episode}_round_{round}.pdf'))
            random_action = env.action_space_sample()
            #import pdb; pdb.set_trace()
            obs, reward, done, info = env.step(random_action)
            round += 1
        print('Episode:', episode, 'Score:', env.score, 'Total Rounds:', round - 1)
        
        if episode == 0:
            scores = []
        scores.append(env.score)
        if episode == num_episodes - 1:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f'Average Score: {avg_score:.2f}, Std_dev: {std_score:.2f}')



if __name__ == '__main__':
    random_baseline(seed=0, num_episodes=3)
