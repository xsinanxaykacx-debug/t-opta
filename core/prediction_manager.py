class PredictionManager:
    def __init__(self, database):
        self.database = database

    def save_prediction(self, match_id, probabilities, decision, model_version, created_at):
        return self.database.save_prediction(match_id, probabilities, decision, model_version, created_at)
