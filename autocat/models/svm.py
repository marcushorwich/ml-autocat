def get_svm_model_v1():
  from ..data.datasets import get_training_data
  from ..data.filters import no_null_StdUnitsShipped_StdNetAmount
  from .pipelines import get_scaled_pipeline_v1
  from . import Model

  # Create feaure matrix and label vector
  X, y = get_training_data('../data/processed/train_2018-08-24.csv', [
      no_null_StdUnitsShipped_StdNetAmount
  ], drop_na=True)
  # Create the pipeline
  pipeline = get_scaled_pipeline_v1(X)
  # Load the model from disk
  model = Model.load('../models/svm-2018-08-24.model', pipeline=pipeline)
  # Fit the model using all of the training data
  model.fit(X, y)

  return (model, X, y)