import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_fn):
        self.attribute_fn = attribute_fn
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        feature = self.attribute_fn(X)
        return pd.concat([X, feature], axis=1)
    
class PandasDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.columns)
        return df
    
def create_feature_adder(attribute_fn):
    """
    Parameters
    ----------
    attribute_fn: function
        A function to apply to the data to map from values x -> y
    Returns
    -------
    numpy.array
        An (n x m+1) matrix where n = # of features, m = # of instances
    """
    feature_adder = CombinedFeatureAdder(attribute_fn)
    return feature_adder

def feature_transactions_per_day(X):
    number_of_transactions_ix, number_of_transaction_days_ix = 8, 9
    transactions_per_day = X.loc[:, 'NumberOfTransactions'] / X.loc[:, 'NumberOfTransactionDays']
    transactions_per_day = transactions_per_day.rename('TransactionsPerDay')
    return transactions_per_day