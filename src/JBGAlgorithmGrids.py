"""Pure search-grid builders shared by algorithm metadata.

Keep persisted AlgorithmGridSearchParams enum members in JBGMeta.
"""

def _pytorch_grid_parameters(activation: str, optimizer: str) -> dict:
    """Six parameter combinations per named variant; do not retune its activation/optimizer identity."""
    rates = (0.0003, 0.001, 0.003) if optimizer == "adam" else (0.01, 0.02, 0.05)
    return {
        "activation": (activation,),
        "optimizer": (optimizer,),
        "learning_rate": rates,
        "max_epochs": (50,),
        "dropout_prob": (0.1,),
        "num_hidden_layers": (2,),
        "hidden_layer_size": (48, 100),
        "train_split": (True,),
    }
