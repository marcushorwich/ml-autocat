def save_model(model, file_path):
  import joblib
  joblib.dump(grid_search_rf.best_estimator_, 'rf.pkl')