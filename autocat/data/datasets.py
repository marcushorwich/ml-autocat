def get_project_data(include_product_data=False):
    from ..settings import TARGET_COLUMN, IDENTIFIER
    
    """ Loads the dataset containing feature and target values

    Parameters
    ----------
    id_col: string
        The name of the record identifier column
    target_column: string
        The name of the target column.  If None then the default name in
        settings.py is used
    Returns
    -------
    pandas.DataFrame
        Pandas dataframe with correct data types
    """
    import pandas as pd

    # Load the GL data
    project_data = pd.read_csv('../data/raw/project-data-2018-02-27.csv', sep='\t')

    if include_product_data:
        # Load the product data
        products = pd.read_csv('../data/raw/upc-product-descriptions.csv', sep='\t')
        # Join products to categories
        project_data = products.merge(project_data, left_on=['UPCCode'], right_on=['UniversalProductCode'])
        project_data = project_data.drop('UPCCode', axis=1)
    
    # Update the labels
    project_data[TARGET_COLUMN] = project_data[TARGET_COLUMN].str.replace(r'Inventory - *', '')
    project_data[IDENTIFIER] = project_data[IDENTIFIER].astype(str)
    
    return project_data

def get_training_data(data_path, filters=None, drop_na=False, target_column=None):
    """ Loads the training data used to train a machine learning model

    Parameters
    ----------
    data_path: string
        The path to the file containing project data.  Expected
        to have a set of identifier, features, and and target value
        records
    filters: 
        A list of filter functions to apply to the training dataset after loading it
    drop_na: boolean
        True if the data should be filtered, false otherwise
    Returns
    -------
    tuple
        A tuple containing the feature and target data, respectively
    """
    import pandas as pd
    from .filters import FeatureFilters
    from ..settings import TARGET_COLUMN

    df = pd.read_csv(data_path)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df.GLCategory

    if filters is not None:
        feature_filter = FeatureFilters(filters)
        X = feature_filter.fit_transform(X)
        y = y.loc[X.index]
    
    return X, y

def get_stratified_train_test_split(data, drop=None):
    """ Performs stratified random sampling on the `data`
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to split into training and test sets
    stratify_on: string
        The column to perform stratified sampling with
    drop: list
        A list of columns to drop
    Returns
    -------
    tuple
        A tuple with (train_df, test_df)
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from ..settings import TARGET_COLUMN, IDENTIFIER
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    if drop is not None:
        df = data.drop(drop + [IDENTIFIER], axis=1)
    else:
        df = data.drop([IDENTIFIER], axis=1)
        
    for train_index, test_index in split.split(df, df[TARGET_COLUMN]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    
    return (strat_train_set, strat_test_set)