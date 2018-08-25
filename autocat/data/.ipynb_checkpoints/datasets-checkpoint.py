DATA_VERSION = '2018-08-04'

def get_project_data(target_column=None):
    from ..settings import TARGET_COLUMN
    
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
    
    if target_column = None:
        target_column = TARGET_COLUMN
    
    # Load the product data
    products = pd.read_csv('../../data/raw/2018-02-27_UPCDescriptions.csv', sep='\t')
    products.head()

    # Load the GL data
    categories = pd.read_csv('../../data/raw/2018-02-27_GLCategoryTrainingData.csv', sep='\t')
    categories.head()

    # Join products to categories
    product_categories = products.merge(categories, left_on=['UPCCode'], right_on=['UniversalProductCode'])
    product_categories = product_categories.drop('UPCCode', axis=1)
    product_categories.head()
    
    # Update the labels
    product_categories[target_column] = product_categories[target_column].str.replace(r'Inventory - *', '')
    product_categories[id_col] = product_categories[id_col].astype(str)
    
    return product_categories

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

    if drop_na:
        feature_filters = FeatureFilters(filters)
        X = feature_filter.fit_transform(X)
        y = y.loc[X.index]
    
    return X, y