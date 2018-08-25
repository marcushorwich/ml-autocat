import pandas as pd
from sklearn.model_selection import GridSearchCV

class Model(object):
    def __init__(self, name, model, pipeline=None):
        self.name = name
        self.model = model
        self.pipeline = pipeline

    def fit(self, X, y):
        X_transformed = self.pipeline.fit_transform(X)
        self.model.fit(X_transformed, y)
        return self
    
    def predict(self, X):
        X_transformed = self.pipeline.fit_transform(X)
        return self.model.predict(X_transformed)

    def get_model_pipeline(self):
        from sklearn.base import clone
        full_pipeline = clone(self.pipeline)
        full_pipeline.steps.append((self.name, self.model))
        return full_pipeline
        

class GridSearchModel(object):
  """ A class used to explore hyperparameters of machine learning
  model using grid search.

  #:todo: this is more like a "TunedModel" - grid search is the implementation

  Parameters
  ----------
  name: string
      The name of a model
  param_grid: dictionary
      A dictionary of (parameter, values) pairs to optimize
  pipeline: object
      A pipeline to apply to the data before fitting the model
  """

  def __init__(self, name, model, param_grid, pipeline):
      from sklearn.base import clone
      self.name = name
      self.model = model
      self.pipeline = pipeline
      self.param_grid = param_grid

  def train(self, X, y, cv_folds, scorer):
      from sklearn.base import clone
      
      model_pipeline = self.get_model_pipeline()

      # Create and execute grid search
      grid_search = GridSearchCV(
          model_pipeline, 
          self.param_grid, 
          cv=cv_folds,
          scoring=scorer, 
          return_train_score=True, 
          n_jobs=-1)

      grid_search.fit(X, y)

      self.model = grid_search.best_estimator_.steps[-1][1]
      self.results = pd.DataFrame(grid_search.cv_results_)

  def get_model_pipeline(self):
      from sklearn.base import clone
      model_pipeline = clone(self.pipeline)
      model_pipeline.steps.append((self.name, self.model))
      return model_pipeline
  
  def save(self, file_path):
      import joblib
      joblib.dump(self.model, file_path)

class ModelEvaluation(object):
    """ A class used to save a model and its evaluation results
    
    Parameters
    ----------
    name: string
        The name of a model
    model: object
        The model used to make predictions
    labels: list
        The original labels or target values
    predictions: list
        Predictions that were made by the model
    score:
        The score of 
        
    Returns
    -------
    numpy.array
        An (n x m+1) matrix where n = # of features, m = # of instances
    """
    __slots__ = ['name', 'model', 'labels', 'predictions', 'score']

    def __init__(self, name, model, labels, predictions, score):
        self.name = name
        self.model = model
        self.labels = labels
        self.predictions = predictions
        self.score = score
    
    def evaluate(self, X, y):
        return ModelEvaluation.evaluate(self, X, y)