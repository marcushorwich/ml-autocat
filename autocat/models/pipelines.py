def get_scaled_pipeline_v1(X):
  """ Returns the first iteration pipeline for this project
    X: pandas.DataFrame
        The feature matrix as a pandas DataFrame with named columns

    Returns
    -------
    sklearn.Pipeline
        Pipeline with the steps required to transform a dataset that looks like 
        `X` to a feature matrix.
  """
  from sklearn.pipeline import Pipeline
  from ..data.features import CombinedFeatureAdder, feature_transactions_per_day, PandasDataFrameTransformer
  from sklearn.preprocessing import StandardScaler
  
  # Create a pipeline to transform the modelling data
  pipeline = Pipeline([
      ('feaure_transactions_per_day', CombinedFeatureAdder(feature_transactions_per_day)),
      ('std_scaler', StandardScaler()),
      ('data_frame', PandasDataFrameTransformer(list(X.columns) + ['TransactionsPerDay']))
  ])
  # print(pipeline.steps)
  return pipeline