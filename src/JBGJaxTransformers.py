import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from JBGTransformers import LongLabelEncoder

# Define the neural network
class MLP(nn.Module):
    hidden_size: int
    num_classes: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)  # Fully connected layer
            x = nn.LayerNorm()(x)             # Apply LayerNorm
            x = nn.relu(x)                    # Activation function
        x = nn.Dense(self.num_classes)(x)     # Output layer (no activation here)
        return x


# Define the FlaxClassifier class
class FlaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=32, num_layers=3, learning_rate=0.001, num_epochs=20, batch_size=32, seed=42, verbose=True):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self.state = None
        self.label_encoder = LongLabelEncoder()

    def _initialize_model(self, input_dim, num_classes):
        self.model = MLP(hidden_size=self.hidden_size, num_classes=num_classes, num_layers=self.num_layers)
        rng = jax.random.PRNGKey(self.seed)
        params = self.model.init(rng, jnp.ones([1, input_dim]))["params"]
        tx = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )
    
    def fit(self, X, y):
        
        self.label_encoder = self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.classes_ = [str(the_class) for the_class in jnp.unique(y_encoded)]
        
        # Convert data to JAX arrays
        X = jnp.array(X)
        y_encoded = jnp.array(y_encoded)
        
        # Initialize the model and training state if not already done
        # Take into account the possibility of an input data dimension mismatch
        if self.state is None or self.state.params['Dense_0']['kernel'].shape[0] != X.shape[1]:
            self._initialize_model(input_dim=X.shape[1], num_classes=len(self.classes_))

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            
            # Shuffle data before creating batches
            rng = jax.random.PRNGKey(self.seed + epoch)  # Ensure different seed per epoch
            perm = jax.random.permutation(rng, len(X))
            X, y_encoded = X[perm], y_encoded[perm]

            # Create batches
            batches = self._create_batches(X, y_encoded)

            # Initialize metrics for the epoch
            epoch_loss = 0.0

            for batch_X, batch_y in batches:
                # Perform one training step
                self.state, batch_loss = self._train_step(self.state, batch_X, batch_y)
                epoch_loss += batch_loss

            # Average loss for the epoch
            epoch_loss /= len(batches)

            # Display progress if verbose is enabled
            if self.verbose:
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {epoch_loss:.4f}")

        return self
    
    def _create_batches(self, X, y):
        
        num_samples = X.shape[0]
        num_batches = int(jnp.ceil(num_samples / self.batch_size))
        batches = []

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, num_samples)
            batch_X = X[start:end]
            batch_y = y[start:end]
            batches.append((batch_X, batch_y))
        
        return batches


    @staticmethod
    @jax.jit
    def _train_step(state, batch, labels):
        
        assert state is not None, "Model state has not been initialized. Call _initialize_model first."
        
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch)
            loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, logits.shape[-1])).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def predict_proba(self, X):
        X = jnp.array(X, dtype=jnp.float32)
        logits = self.model.apply({"params": self.state.params}, X)
        probabilities = jax.nn.softmax(logits, axis=-1)
        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def main():

    # Check platform
    print(f"Available JAX devices: {jax.devices()}")
    jax.config.update("jax_platform_name", "gpu")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_classes=3, random_state=42)
    y = np.array(["Class " + str(y[i]) for i in range(y.shape[0])])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Flax classifier
    clf = FlaxClassifier(hidden_size=32, num_layers=3, learning_rate=0.001, num_epochs=20, batch_size=32, verbose=True)
    clf.fit(X_train, y_train)

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Start main
if __name__ == "__main__":
    main()
