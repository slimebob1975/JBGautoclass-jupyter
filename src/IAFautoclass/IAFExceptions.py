class DatasetException(Exception):
    def __init__(self, message):
        super().__init__(f"DatasetException: {message}")

class HandlerException(Exception):
    def __init__(self, message):
        super().__init__(f"HandlerException: {message}")

class ModelException(Exception):
    def __init__(self, message):
        super().__init__(f"ModelException: {message}")

class PredictionsException(Exception):
    def __init__(self, message):
        super().__init__(f"PredictionsException: {message}")