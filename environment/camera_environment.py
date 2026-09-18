from camera.CameraController import CameraController
from processing.image_analyzer import ImageAnalyzer
from agent.RLAgent import RLAgent


class CameraEnvironment:

    def __init__(self):
        self.camera = CameraController()
        self.analyzer = ImageAnalyzer()

        self.actions = [
            {"exposure": 20000},
            {"exposure": 30000},
            {"exposure": 40000},
        ]

        self.agent = RLAgent(self.actions)
        self.step_count = 0

    def step(self, training = True):
        self.step_count += 1

        image = self.camera.capture_image()

        if image is None:
            return {
                "step": self.step_count,
                "error": "image_capture_failed"
            }

        features = self.analyzer.extract_features(image)

        defect_prob = self.analyzer.compute_defect_probability(
            features
        )

        action = self.agent.choose_action(features)

        self.camera.set_parameters(**action)

        image2 = self.camera.capture_image()

        if image2 is None:
            return {
                "step": self.step_count,
                "error": "image_capture_failed_after_action",
                "action": action
            }

        features2 = self.analyzer.extract_features(image2)

        defect_prob2 = self.analyzer.compute_defect_probability(
            features2
        )

        reward = self._calculate_reward(
            defect_prob,
            defect_prob2
        )

        if training:
         self.agent.learn(
        features,
        action,
        reward,
        next_state=features2
    )

        return {
            "step": self.step_count,
            "before_defect_prob": defect_prob,
            "after_defect_prob": defect_prob2,
            "reward": reward,
            "action": action,
            "features_before": features,
            "features_after": features2,
            
            "image_before": image, 
            "image_after": image2,
        }

    def _calculate_reward(self, before, after):
        return (before - after) * 10

    def reset(self):
        self.step_count = 0

    def close(self):
        self.camera.close()
