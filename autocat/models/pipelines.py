def get_scaled_pipeline_v1():
  from sklearn.pipeline import Pipeline
  from ...data.features import CombinedFeatureAdder, feature_transactions_per_day, PandasDataFrameTransformer
  from sklearn.preprocessing import StandardScaler
  
  # Create a pipeline to transform the modelling data
  pipeline = Pipeline([
      ('feaure_transactions_per_day', CombinedFeatureAdder(feature_transactions_per_day)),
      ('std_scaler', StandardScaler()),
      ('data_frame', PandasDataFrameTransformer(list(X.columns) + ['TransactionsPerDay']))
  ])
  print(pipeline.steps)

get_scaled_pipeline_v1()