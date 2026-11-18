class MockModel:
    version = "mock-baseline-1"

    def predict(self, features):
        return {"probabilities": {"home": 0.34, "draw": 0.32, "away": 0.34},
                "confidence": None, "decision": "SIMULATION_ONLY", "calibrated": False,
                "model_version": self.version,
                "warning": "MOCK output is a simulation, not a betting recommendation."}
