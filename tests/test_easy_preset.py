import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gym-dragon"))

try:
    from gym_dragon.envs import MiniDragonEnv
    from gym_dragon.wrappers import MiniObs
    from random_baseline import PRESETS
except ModuleNotFoundError:
    MiniDragonEnv = MiniObs = PRESETS = None


@unittest.skipIf(MiniDragonEnv is None, "project runtime dependencies are not installed")
class EasyPresetTests(unittest.TestCase):
    def test_places_three_unique_three_phase_bombs(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                env = MiniDragonEnv(
                    valid_nodes=PRESETS["easy"],
                    obs_wrapper=MiniObs,
                    include_chained_bombs=False,
                    include_fire_bombs=False,
                    include_fuse_bombs=False,
                )
                env.seed(seed)
                env.reset(
                    csv_path=None,
                    num_bombs_per_region=3,
                    bomb_sequence_length=3,
                )

                self.assertEqual(len(env._bombs), 3)
                self.assertEqual(len({bomb.location for bomb in env._bombs}), 3)
                self.assertTrue(all(bomb.num_stages == 3 for bomb in env._bombs))


if __name__ == "__main__":
    unittest.main()
