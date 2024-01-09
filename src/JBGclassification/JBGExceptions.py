class DatasetException(Exception):
    def __init__(self, message):
        super().__init__(f"DatasetException: {message}")

class HandlerException(Exception):
    def __init__(self, message):
        super().__init__(f"HandlerException: {message}")

class ModelException(Exception):
    def __init__(self, message):
        super().__init__(f"ModelException: {message}")

class PipelineException(Exception):
    def __init__(self, message):
        super().__init__(f"PipelineException: {message}")

class UnstableModelException(Exception):
    def __init__(self, message):
        super().__init__(f"UnstableModelException: {message}")

class PredictionsException(Exception):
    def __init__(self, message):
        super().__init__(f"PredictionsException: {message}")

class DataLayerException(Exception):
    def __init__(self, message):
        super().__init__(f"DataLayerException: {message}")

class ConfigException(Exception):
    def __init__(self, message):
        super().__init__(f"ConfigException: {message}")

class GuiWidgetsException(Exception):
    def __init__(self, message):
        super().__init__(f"GuiWidgetsException: {message}")
        
class SQLException(Exception):
    def __init__(self, message):
        super().__init__(f"SQLException: {message}")

class ODBCDriverException(Exception):
    def __init__(self, message):
        super().__init__(f"ODBCDriverException: {message}")