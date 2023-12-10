import optuna
import itertools
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

def calculate_metrics(model, interactions, item_features, user_features, k):
    precision = precision_at_k(model, interactions, item_features=item_features, user_features=user_features, k=k).mean()
    recall = recall_at_k(model, interactions, item_features=item_features, user_features=user_features, k=k).mean()
    auc = auc_score(model, interactions, item_features=item_features, user_features=user_features).mean()
    return precision, recall, auc

# default number of recommendations
K = 10
# percentage of data used for testing
TEST_PERCENTAGE = 0.2
# no of threads to fit model
NO_THREADS = 32
checkpoint = 'lightFM_hybrid'
# seed for pseudonumber generations
SEED = 42

data = pd.read_csv('data/interim/100k_clean.csv', index_col=0)
data = data[data['rating'] > 3]

def item_feature_generator(data):
    for i, row in data.iterrows():
        features = row['genre'].split(', ')
        yield (row['item id'], features)

def user_feature_generator(data, user_cols):
    for i, row in data.iterrows():
        features = [row[column] for column in user_cols]
        yield (row['user id'], features)

def objective(trial):
    item_cols = ['genre']
    user_cols = ['gender', 'occupation', 'age_group']
    global data
    '''
    Parameters
    no_components (int, optional) - the dimensionality of the feature latent embeddings.

    k (int, optional) - for k-OS training, the k-th positive example will be selected from the n positive examples sampled for every user.

    n (int, optional) - for k-OS training, maximum number of positives sampled for each update.

    learning_schedule (string, optional) - one of ('adagrad', 'adadelta').

    loss (string, optional) - one of ('logistic', 'bpr', 'warp', 'warp-kos'): the loss function.

    learning_rate (float, optional) - initial learning rate for the adagrad learning schedule.

    rho (float, optional) - moving average coefficient for the adadelta learning schedule.

    epsilon (float, optional) - conditioning parameter for the adadelta learning schedule.

    item_alpha (float, optional) - L2 penalty on item features. Tip: setting this number too high can slow down training. One good way to check is if the final weights in the embeddings turned out to be mostly zero. The same idea applies to the user_alpha parameter.

    user_alpha (float, optional) - L2 penalty on user features.

    max_sampled (int, optional) - maximum number of negative samples used during WARP fitting. It requires a lot of sampling to find negative triplets for users that are already well represented by the model; this can lead to very long training times and overfitting. Setting this to a higher number will generally lead to longer training times, but may in some cases improve accuracy.

    random_state (int seed, RandomState instance, or None) - The seed of the pseudo random number generator to use when shuffling the data and initializing the parameters.
    '''
    loss = 'warp'
    LEARNING_RATE = trial.suggest_categorical('learning_rate', [1, 0.5, 0.1, 0.01, 0.001, 0.0001])
    NO_COMPONENTS = trial.suggest_categorical('no_components', [16, 64, 128, 256, 512])
    NO_EPOCHS = 100
    ITEM_ALPHA = trial.suggest_categorical('item_alpha', [1e-6, 1e-5, 1e-4, 1e-3])
    USER_ALPHA = trial.suggest_categorical('user_alpha', [1e-6, 1e-5, 1e-4, 1e-3])
    

    all_item_features = list(set(itertools.chain.from_iterable([x.split(', ') for x in data['genre']])))
    all_user_features = np.concatenate([data[col].unique() for col in user_cols]).tolist()

    dataset = Dataset()
    dataset.fit(
        users=data['user id'], 
        items=data['item id'],
        user_features=all_user_features,
        item_features=all_item_features
    )

    (interactions, weights) = dataset.build_interactions(zip(data['user id'], data['item id']))

    item_features = dataset.build_item_features((item_id, item_feature) for item_id, item_feature in item_feature_generator(data))
    user_features = dataset.build_user_features((user_id, user_feature) for user_id, user_feature in user_feature_generator(data, user_cols))
    
    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model = LightFM(
        loss=loss,
        learning_schedule='adagrad',
        no_components=NO_COMPONENTS,
        learning_rate=LEARNING_RATE,
        item_alpha=ITEM_ALPHA,
        user_alpha=USER_ALPHA,
        random_state=np.random.RandomState(SEED)
    )

    model.fit(
        interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=NO_EPOCHS,
        num_threads=NO_THREADS,
        verbose=True,
    )
    
    test_precision, test_recall, test_auc = calculate_metrics(model, test_interactions, item_features, user_features, K)

    
    return test_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

trial = study.best_trial

print('ROC AUC score: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))