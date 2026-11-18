from core.model_manager import ModelManager

class PredictionPipeline:
    def __init__(self, environment="MOCK"):
        self.model = ModelManager(environment).load()

    def predict(self, features):
        return self.model.predict(features)
