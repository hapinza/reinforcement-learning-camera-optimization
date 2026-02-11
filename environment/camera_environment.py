from camera.camera_controller import camera_controller
from processing.image_analyzer import ImageAnalyzer 
from agent.rl_agent import rl_agent


class CameraEnvironment:

    def __init__(self):
        self.camera = camera_controller()
        self.analyzer = ImageAnalyzer()


    
        actions = [
            {"exposure": 1000},
            {"exposure": 2000},
            {"exposure": 3000}
        ]

        self.agent = rl_agent(actions)
        self.step_count = 0

        

    def step(self):

        self.step_count += 1

        image = self.camera.capture_image()
        if image is None:
            return {"step": self.step_count, "error": "image_capture_failed"}

        features = self.analyzer.extract_features(image)

        defect_prob = self.analyzer.compute_defect_probability(features)

        action = self.agent.choose_action(features)

        self.camera.self_parameter(**action)

        image2 = self.camera.capture_image()
        if image2 is None:
            return {"step": self.step_count, "error": "image_capture_failed_after_action", "action": action}


        features2 = self.analyzer.extract_features(image2)

        defect_prob2 = self.analyzer.compute_defect_probability(features2)

        reward = self._calculate_reward(defect_prob2)

        self.agent.learn(features, action, reward, next_state= features2)

        return {
            "step": self.step_count,
            "defect_prob": defect_prob,
            "reward": reward,
            "action": action
        }

       
    
    

    def _calculate_reward(self, defect_prob):
        return defect_prob*10
    

    def reset(self):
        self.step_count = 0
        print("Environment reset")

    
    def close(self):
        self.camera.close()
        print("Environment closed")

        