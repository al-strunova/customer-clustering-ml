import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    A class used to preprocess the e-commerce dataset.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer (no fitting needed in this case).

    transform(X)
        Applies various preprocessing steps to the data.
    """

    def __init__(self):
        """
        Initializes the transformer.
        """

        # Set default values for some statistics to be used later
        self.column_stats = {}

        # Initialize scaler and pca to be used for stockCode transformation
        self.stockcode_scaler = StandardScaler()
        self.pca = PCA(n_components=0.80)

        self.top_countries = []

    def fit(self, X, y=None):
        """
        Fits the transformer.

        Parameters:
        X (DataFrame): Original data.
        y (array-like): Target values (not used).

        Returns:
        self: Returns the fitted transformer.
        """

        # Creating a copy of X to work with
        X_copy = X.copy()
        X_copy.dropna(subset=['CustomerID'], inplace=True)

        # Create Sales column
        X_copy['Sales'] = X_copy['Quantity'] * X_copy['UnitPrice']

        # Outliers Statistics
        self.column_stats = {}
        for column in ['Quantity', 'UnitPrice', 'Sales']:
            self.column_stats[column] = {'mean': X_copy[column].mean(), 'std': X_copy[column].std()}

        # Determine the top countries based on the training data and save them
        self.top_countries = X_copy['Country'].value_counts().head(5).index

        # PCA for StockCode
        # Fitting the scaler and PCA for StockCode
        stockcode_dummies = pd.get_dummies(X_copy['StockCode'], prefix='StockCode')
        stockcode_dummies['CustomerID'] = X_copy['CustomerID']  # Attaching CustomerID to stockcode_dummies
        item_data = stockcode_dummies.groupby('CustomerID').sum()

        # Scaling and fitting PCA
        item_data_scaled = self.stockcode_scaler.fit_transform(item_data)
        self.pca.fit(item_data_scaled)

        return self

    def transform(self, X):
        """
        Transforms the data by applying all preprocessing steps.

        Parameters:
        X (DataFrame): Original data.

        Returns:
        DataFrame: Transformed data.
        DataFrame: Customer aggregated data.
        """
        # Creating a copy of X to work with
        X_copy = X.copy()

        # Dropping rows where 'CustomerID' is missing
        X_copy = X_copy.dropna(subset=['CustomerID'])

        # Changing CustomerID type to int
        X_copy['CustomerID'] = X_copy.CustomerID.astype(int)

        # Creating new feature
        X_copy['Sales'] = X_copy['Quantity'] * X_copy['UnitPrice']

        # Handling Outliers using pre-calculated statistics
        for col in ['Quantity', 'UnitPrice', 'Sales']:
            mean, std = self.column_stats[col]['mean'], self.column_stats[col]['std']
            X_copy[col], X_copy[f'Outlier_{col}'] = self.handle_outliers(X_copy[col], mean, std)

        # Processing datetime
        X_copy['InvoiceDate'] = pd.to_datetime(X_copy['InvoiceDate'])  # Ensure the InvoiceDate is in datetime format
        max_invoice_date = X_copy['InvoiceDate'].max()  # Get the maximum invoice date in the dataset
        X_copy['Recency'] = (max_invoice_date - X_copy.groupby('CustomerID')['InvoiceDate'].transform('max')).dt.days

        # Grouping countries
        # Marking countries that are not in the top 5 as 'Other'
        X_copy['Country_aggregated'] = X_copy['Country'].apply(lambda x: x if x in self.top_countries else 'Other')

        # Processing StockCode
        stockcode_pca = self.process_stockcode(X_copy)

        # Customer aggregate
        customer_df = self.customer_aggregate(X_copy, stockcode_pca)

        return X_copy, customer_df

    def handle_outliers(self, column, mean, std):
        """
        Handles outliers in a column by capping.

        Parameters:
        column (Series): Pandas Series containing column values.

        Returns:
        Series: Transformed column.
        """
        # Calculating z-scores based on provided mean and std
        z_scores = np.abs((column - mean) / std)

        # Flag outliers
        is_outlier = (z_scores > 3).astype(int)

        # Capping values
        upper_cap = mean + 3 * std
        lower_cap = mean - 3 * std
        capped_column = np.where(column > upper_cap, upper_cap, column)
        capped_column = np.where(capped_column < lower_cap, lower_cap, capped_column)

        return capped_column, is_outlier

    def customer_aggregate(self, df, stockcode_pca):
        """
        This function aggregates the transactional data at the customer level to create features that will be
        useful for customer segmentation. The features created for each customer are as follows:

        - total_transactions: The total number of unique transactions made by a customer.

        - total_products: The total number of products purchased by a customer across all transactions.

        - total_unique_products: The total number of unique products purchased by a customer across all transactions.

        - total_sales: The total sales value generated by a customer across all transactions.

        - avg_product_value: The average value of the products purchased by a customer.

        - outlier_flag: A binary flag indicating whether a customer has made any transactions considered as outliers
          based on sales value.

        - avg_cart_value: The average sales value per transaction for a customer.

        - min_cart_value: The minimum sales value per transaction for a customer.

        - max_cart_value: The maximum sales value per transaction for a customer.

        - country_{CountryName}: Binary flags indicating the presence of transactions from specific countries. The
          top 5 countries in terms of the number of transactions are kept as separate flags, and the rest are grouped
          under 'Other'.

        Parameters:
        df (DataFrame): A DataFrame containing the original transaction data. The DataFrame is expected to contain
        at least the following columns: ['CustomerID', 'InvoiceNo', 'StockCode', 'Sales', 'Country'].
        stockcode_pca (DataFrame): Customer-level PCA features from stock codes.

        Returns:
        DataFrame: A DataFrame where each row corresponds to a unique CustomerID and columns correspond to the
        aggregated features as described above.
        """
        # Group by CustomerID aggregate

        invoice_data = df.groupby('CustomerID').InvoiceNo.agg(['nunique'])
        invoice_data.columns = ['total_transactions']

        product_data = df.groupby('CustomerID').StockCode.agg(['count', 'nunique'])
        product_data.columns = ['total_products', 'total_unique_products']

        sales_data = df.groupby('CustomerID').Sales.agg(['sum', 'mean'])
        sales_data.columns = ['total_sales', 'avg_product_value']

        outlier_data = df.groupby('CustomerID')[['Outlier_Quantity', 'Outlier_UnitPrice', 'Outlier_Sales']].agg(['sum'])
        outlier_data.columns = ['Outlier_Quantity', 'Outlier_UnitPrice', 'Outlier_Sales']
        outlier_data['pct_quantity_outliers'] = outlier_data['Outlier_Quantity'] / product_data['total_products']
        outlier_data['pct_price_outliers'] = outlier_data['Outlier_UnitPrice'] / product_data['total_products']
        outlier_data['pct_sales_outliers'] = outlier_data['Outlier_Sales'] / product_data['total_products']

        # Group by CustomerID cart

        cart_data = df.groupby(['CustomerID', 'InvoiceNo']).Sales.agg(['sum'])
        cart_data.columns = ['cart_value']
        cart_data.reset_index(inplace=True)

        agg_cart_data = cart_data.groupby('CustomerID').cart_value.agg(['mean', 'min', 'max'])
        agg_cart_data.columns = ['avg_cart_value', 'min_cart_value', 'max_cart_value']

        # Country aggregation
        country_data = df.groupby(['CustomerID', 'Country_aggregated']).InvoiceNo.count().unstack(fill_value=0)
        country_data.columns = ['country_' + str(col) for col in country_data.columns]

        # Aggregating Recency: the minimum recency per customer as it will be same for all transactions of a customer
        recency_data = df.groupby('CustomerID')['Recency'].min()
        recency_data.columns = ['Recency']

        # Join all new datasets together to a customer DF
        joined_df = invoice_data.join([product_data,
                                       sales_data,
                                       outlier_data,
                                       agg_cart_data,
                                       country_data,
                                       recency_data,
                                       stockcode_pca],
                                      how='left').fillna(0)
        return joined_df

    def process_stockcode(self, df):
        """
        Processes StockCode using one-hot encoding and PCA.

        Parameters:
        df (DataFrame): Original data.

        Returns:
        DataFrame: Processed data.
        """
        # Step 1: One-hot encoding
        stockcode_dummies = pd.get_dummies(df['StockCode'], prefix='StockCode')
        stockcode_dummies['CustomerID'] = df['CustomerID']

        # Step 2: Grouping by CustomerID
        item_data = stockcode_dummies.groupby('CustomerID').sum()

        # Step 3: Scaling
        item_data_scaled = self.stockcode_scaler.transform(item_data)

        # Step 4: Applying PCA
        pca_features = self.pca.transform(item_data_scaled)

        # Step 5: Creating DataFrame for PCA features
        items_pca = pd.DataFrame(pca_features, columns=['PC{}'.format(i + 1) for i in range(pca_features.shape[1])])
        items_pca.index = item_data.index  # Setting CustomerID as the index

        return items_pca
