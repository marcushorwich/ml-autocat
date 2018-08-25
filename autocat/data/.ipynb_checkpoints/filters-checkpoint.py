from sklearn.base import BaseEstimator, TransformerMixin

class FeatureFilters(BaseEstimator, TransformerMixin):
    def __init__(self, filter_fns):
        self.filter_fns = filter_fns
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        df = X.copy()
        
        for f in self.filter_fns:
            ix = f(df)
            df = df.loc[ix, :]
        return df
    
no_null_StdUnitsShipped_StdNetAmount = lambda df: ~(df.StdUnitsShipped.isnull()) | ~(df.StdNetAmount.isnull())