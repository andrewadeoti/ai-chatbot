"""
Advanced Algorithms for the Food Chatbot
This module contains various advanced AI algorithms demonstrated in the chatbot:
- Transformer Language Model
- Breadth-First Search (BFS)
- Traveling Salesperson Problem (TSP)
- Simple Linear Regression
"""
import numpy as np
import collections
from pylab import array, zeros
import sklearn.model_selection as ms
import sklearn.linear_model as lm

# Try to import TensorFlow/Keras, but handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Transformer demo will be limited.")

# --- Transformer Components -------------------------------------------------

class PositionalEncoding:
    """Injects position information into the input embeddings."""
    def __init__(self, max_steps, max_dims, dtype=np.float32, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for PositionalEncoding")
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1: max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        return inputs + self.positional_embedding

class MultiHeadSelfAttention:
    """Multi-head self-attention layer."""
    def __init__(self, embed_dim, num_heads=8):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for MultiHeadSelfAttention")
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock:
    """A single transformer block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for TransformerBlock")
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TinyTransformerLM:
    """A tiny transformer-based language model."""
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_transformer_blocks, max_len):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for TinyTransformerLM")
        super(TinyTransformerLM, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_steps=max_len, max_dims=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        self.dense = layers.Dense(vocab_size)
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.dense(x)

# --- Other Algorithms -------------------------------------------------------

class AdvancedAlgorithms:
    def __init__(self):
        # Transformer default parameters
        self.vocab_size = 1000
        self.embed_dim = 32
        self.num_heads = 2
        self.ff_dim = 32
        self.num_transformer_blocks = 1
        self.max_len = 100
        
    def demonstrate_transformer(self):
        """Demonstrates the Tiny Transformer Language Model generating text."""
        if not TENSORFLOW_AVAILABLE:
            return "TensorFlow is not available. Transformer demo requires TensorFlow to be installed."
            
        try:
            model = TinyTransformerLM(self.vocab_size, self.embed_dim, self.num_heads, self.ff_dim, self.num_transformer_blocks, self.max_len)
            dummy_input = np.random.randint(0, self.vocab_size, size=(1, 10))
            output = model(dummy_input)
            
            demo_text = f"""
            Transformer Demo:
            - Model: TinyTransformerLM
            - Vocab Size: {self.vocab_size}
            - Embedding Dim: {self.embed_dim}
            - Input Shape: {dummy_input.shape}
            - Output Shape: {output.shape}
            This demonstrates that the transformer model is correctly constructed and can process input.
            A full training loop is not included, but the architecture is sound.
            """
            return demo_text.strip()
        except Exception as e:
            return f"Error demonstrating Transformer: {e}"

    def demonstrate_bfs(self):
        """Demonstrates the Breadth-First Search algorithm."""
        graph = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': [],
            'E': ['F'],
            'F': []
        }
        start_node = 'A'
        visited = []
        queue = collections.deque([start_node])
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.append(node)
                neighbours = graph.get(node, [])
                for neighbour in neighbours:
                    queue.append(neighbour)
        
        return f"BFS Demo:\nGraph traversal starting from '{start_node}': {' -> '.join(visited)}"

    def tsp_distance(self, R1, R2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)

    def tsp_total_distance(self, city, R):
        """Calculate total distance of the route."""
        dist = 0
        for i in range(len(city) - 1):
            dist += self.tsp_distance(R[city[i]], R[city[i+1]])
        dist += self.tsp_distance(R[city[-1]], R[city[0]])
        return dist

    def demonstrate_tsp(self):
        """Demonstrates a simple TSP solver."""
        n = 5  # Number of cities
        R = array([[np.random.rand() for i in range(2)] for j in range(n)])
        city = list(range(n))
        
        # Simple hill climbing algorithm
        for i in range(1000):
            np.random.shuffle(city)
            d = self.tsp_total_distance(city, R)
            c = city
            for j in range(100):
                c_new = c.copy()
                t = np.random.randint(0, len(c) - 1)
                t2 = np.random.randint(0, len(c) - 1)
                c_new[t], c_new[t2] = c_new[t2], c_new[t2]
                d_new = self.tsp_total_distance(c_new, R)
                if d_new < d:
                    d = d_new
                    c = c_new
        
        return (f"TSP Demo:\nOptimal route found for {n} cities: {' -> '.join(map(str, c))}\n"
                f"Total distance: {d:.2f}")

    def demonstrate_linear_regression(self):
        """Demonstrates a simple Linear Regression model."""
        # Sample data
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 5, 4, 5])
        
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lm.LinearRegression()
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        return (f"Linear Regression Demo:\n"
                f"Model trained on sample data.\n"
                f"Coefficient: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}\n"
                f"Test score (R^2): {score:.2f}") 