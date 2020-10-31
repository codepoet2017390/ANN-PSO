import functools
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
from sklearn.impute import SimpleImputer


import pso
import ann


def dim_weights(shape):
    dim = 0
    for i in range(len(shape)-1):
        dim = dim + (shape[i] + 1) * shape[i+1]
    return dim

def weights_to_vector(weights):
    w = np.asarray([])
    for i in range(len(weights)):
        v = weights[i].flatten()
        w = np.append(w, v)
    return w

def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape)-1):
        r = shape[i+1]
        c = shape[i] + 1
        idx_min = idx
        idx_max = idx + r*c
        W = vector[idx_min:idx_max].reshape(r,c)
        weights.append(W)
    return weights

def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for w in weights:
        weights = vector_to_weights(w, shape)
        nn = ann.MultiLayerPerceptron(shape, weights=weights)
        y_pred = nn.run(X)
        mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
    return mse

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)   

def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))


# Load MNIST digits from sklearn
num_classes = 2

dataset = pd.read_csv('jm1_csv.csv')

clean_dataset(dataset)

X_raw = dataset.iloc[:, :-1].values
y_raw = dataset.iloc[:, -1].values

print(type(dataset))
print(np.any(np.isnan(X_raw)))
X, X_test, y, y_test = sklearn.model_selection.train_test_split(X_raw, y_raw, test_size = 0.2, random_state = 1)

print(len(X))

num_inputs = X.shape[1]

y_true = np.zeros((len(y), num_classes))
for i in range(len(y)):
    y_true[i, y[i]] = 1

y_test_true = np.zeros((len(y_test), num_classes))
for i in range(len(y_test)):
    y_test_true[i, y_test[i]] = 1

# Set up
shape = (num_inputs, 64, 32, 16, num_classes)

cost_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

swarm = pso.ParticleSwarm(cost_func, num_dimensions=dim_weights(shape), num_particles=30)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
while swarm.best_score>1e-6 and i<500:
    swarm._update()
    i = i+1
    if swarm.best_score < best_scores[-1][1]:
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])

# Test...
best_weights = vector_to_weights(swarm.g, shape)
best_nn = ann.MultiLayerPerceptron(shape, weights=best_weights)
y_test_pred = np.round(best_nn.run(X_test))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))