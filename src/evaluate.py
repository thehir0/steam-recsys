from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

def calculate_metrics(model, interactions, item_features, user_features, k):
    precision = precision_at_k(model, interactions, item_features=item_features, user_features=user_features, k=k).mean()
    recall = recall_at_k(model, interactions, item_features=item_features, user_features=user_features, k=k).mean()
    auc = auc_score(model, interactions, item_features=item_features, user_features=user_features).mean()
    return precision, recall, auc