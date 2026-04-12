import json
import unittest
from models import AQIAction, AQIObservation

class TestSerialization(unittest.TestCase):
    def test_action_serialization(self):
        # Test single action
        a1 = AQIAction(action_type="restrict_traffic", level=2, city="Delhi")
        data = a1.model_dump()
        self.assertEqual(data["action_type"], "restrict_traffic")
        
        # Test multi-action bundle
        a2 = AQIAction(
            actions=[
                {"action_type": "close_schools", "level": 1},
                {"action_type": "mandate_wfh", "level": 1}
            ],
            city="Delhi"
        )
        data = a2.model_dump()
        self.assertEqual(len(data["actions"]), 2)
        
        # Test GRAP stage
        a3 = AQIAction(grap_stage=3, city="Delhi")
        data = a3.model_dump()
        self.assertEqual(data["grap_stage"], 3)

    def test_observation_roundtrip(self):
        obs = AQIObservation(
            day=5,
            city="Delhi",
            pm25_today=350.0,
            pm25_post_intervention=310.0,
            estimated_hospital_admissions=120,
            public_sentiment=0.45,
            cities_data={"lucknow": {"pm25_today": 280.0, "grap_stage": 3}}
        )
        json_str = obs.model_dump_json()
        obs_decoded = AQIObservation.model_validate_json(json_str)
        self.assertEqual(obs_decoded.day, 5)
        self.assertEqual(obs_decoded.city, "Delhi")
        self.assertEqual(obs_decoded.estimated_hospital_admissions, 120)
        self.assertEqual(obs_decoded.cities_data["lucknow"]["grap_stage"], 3)

if __name__ == "__main__":
    unittest.main()
