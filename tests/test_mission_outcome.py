import unittest

from utils.utils import build_bomb_mission_outcome


class MissionOutcomeTests(unittest.TestCase):
    def test_all_defused_is_accomplished_with_dynamic_max_score(self):
        outcome = build_bomb_mission_outcome(
            ["defused", "defused", "defused"],
            [1, 2, 3],
            "environment_terminated",
        )

        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["status"], "accomplished")
        self.assertEqual(outcome["reason_code"], "all_objectives_completed")
        self.assertEqual(outcome["objectives_completed"], 3)
        self.assertEqual(outcome["max_score"], 60)

    def test_round_limit_failure(self):
        outcome = build_bomb_mission_outcome(
            ["defused", "active"], [2, 3], "round_limit_reached"
        )

        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["reason_code"], "round_limit_reached")
        self.assertEqual(outcome["objectives_completed"], 1)

    def test_mission_time_failure(self):
        outcome = build_bomb_mission_outcome(
            ["inactive"], [3], "mission_time_expired"
        )

        self.assertEqual(outcome["reason_code"], "mission_time_expired")

    def test_explosion_takes_precedence_over_stop_reason(self):
        outcome = build_bomb_mission_outcome(
            ["defused", "exploded"], [1, 2], "round_limit_reached"
        )

        self.assertEqual(outcome["reason_code"], "bombs_exploded")
        self.assertEqual(outcome["details"]["bombs_exploded"], 1)


if __name__ == "__main__":
    unittest.main()
