from camera.CameraController import CameraController
from processing.image_analyzer import ImageAnalyzer
from agent.RLAgent import RLAgent


class CameraEnvironment:

    def __init__(self):
        self.camera = CameraController()
        self.analyzer = ImageAnalyzer()

        self.actions = [
            {"exposure": 500},
            {"exposure": 1000},
            {"exposure": 2000},
            {"exposure": 5000},
            {"exposure": 10000},
        ]

        self.agent = RLAgent(self.actions)
        self.step_count = 0

    def step(self, training=True):
        self.step_count += 1

        # --------------------------------------------------
        # 1. Capture current state
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 2. Get current state + Q-values BEFORE action
        # --------------------------------------------------
        state_key = self.agent.state_to_key(features)

        q_values_before = self._get_q_values(state_key)

        # --------------------------------------------------
        # 3. RL chooses exposure
        # --------------------------------------------------
        action = self.agent.choose_action(features)

        # --------------------------------------------------
        # 4. Apply new exposure
        # --------------------------------------------------
        self.camera.set_parameters(**action)

        # --------------------------------------------------
        # 5. Discard stale frame
        #
        # We experimentally observed that the first frame
        # after an exposure change still reflects the
        # previous exposure value.
        # --------------------------------------------------
        discarded_image = self.camera.capture_image()

        if discarded_image is None:
            return {
                "step": self.step_count,
                "error": "discard_frame_failed",
                "action": action
            }

        # --------------------------------------------------
        # 6. Capture image with the NEW exposure
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 7. Calculate reward
        # Lower proxy = better
        # --------------------------------------------------
        reward = self._calculate_reward(
            defect_prob,
            defect_prob2
        )

        # --------------------------------------------------
        # 8. Q-learning update
        # --------------------------------------------------
        if training:
            self.agent.learn(
                features,
                action,
                reward,
                next_state=features2
            )

        # Q-values after this learning step
        q_values_after = self._get_q_values(state_key)

        return {
            "step": self.step_count,

            "state_key": state_key,

            "before_defect_prob": defect_prob,
            "after_defect_prob": defect_prob2,

            "reward": reward,
            "action": action,

            "features_before": features,
            "features_after": features2,

            "q_values_before": q_values_before,
            "q_values_after": q_values_after,

            "image_before": image,
            "image_after": image2,
        }

    def _get_q_values(self, state_key):
        """
        Return Q-values for every exposure action.

        If this state has never been seen before,
        all Q-values start at 0.
        """

        if state_key not in self.agent.q_table:
            return {
                action["exposure"]: 0.0
                for action in self.actions
            }

        state_q = self.agent.q_table[state_key]

        return {
            action["exposure"]: state_q.get(i, 0.0)
            for i, action in enumerate(self.actions)
        }

    def _calculate_reward(self, before, after):
        return (before - after) * 10

    def reset(self):
        self.step_count = 0

    def close(self):
        self.camera.close()
