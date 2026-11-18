from models.mock.mock_model import MockModel

class ModelManager:
    def __init__(self, environment="MOCK", model_path=None):
        self.environment = environment.upper()
        self.model_path = model_path

    def load(self):
        if self.environment == "MOCK":
            return MockModel()
        if self.environment in {"STAGING", "PRODUCTION"}:
            raise RuntimeError(f"No approved model configured for {self.environment}; refusing mock fallback.")
        raise ValueError(f"Unknown environment: {self.environment}")
