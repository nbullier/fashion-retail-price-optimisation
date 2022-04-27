from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import (datetime as dt, timedelta)
import pytrends
from pytrends.request import TrendReq
from mip import Model, xsum, maximize, BINARY, OptimizationStatus
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score, mean_absolute_error)
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

class PriceOptim:
    def __init__(self):
        self.path_root = Path('..')
        self.path_data = Path(self.path_root / 'data')

    def data_process_cww(self, prod_data, trans_data):
        '''
        Preparing and cleaning data

            Parameters
            ----------
            prod_data: pandas DataFrame
                table of products
            trans_data: pandas DataFrame
                table of transactions

            Returns
            -------
            merged_data_pruned_sd_knits:
                pandas dataframe of merged transactions for subsequent data synthesis
        '''

        # ### load and merge data
        # prod_data = pd.read_excel(Path(self.path_data / 'raw/Product_data.xlsx'), sheet_name="Data")
        # trans_data = pd.read_excel(Path(self.path_data / 'raw/Transaction_data.xlsx'), sheet_name="Data")

        def merge_product_transactions(product_data=prod_data, transaction_data=trans_data):
            '''
            Returns a merged dataframe of product data and transaction data.
                Parameters:
                    product_data (df): A dataframe containing product information
                    transaction_data (df): A dataframe containing transaction information.
                Returns:
                    merged_data (df): A dataframe containing prodcut and transaction data, merged on sku.
            '''
            merged_data = pd.merge(prod_data, trans_data, on="sku", how="right")
            return merged_data

        merged_data = merge_product_transactions(product_data=prod_data, transaction_data=trans_data)

        # ### remove characters from the p_id_x / p_id_y
        def fix_pid_cols(df):
            '''
            Removes extra characters from product ID (p_id) column.
                Parameters:
                    df (df): Dataframe from which to remove extra characters from p_id column.
                Returns:
                    df (df): A dataframe with extra characters removed from the product ID (p_id) column.
            '''
            merged_data['p_id'] = merged_data['p_id_y'].astype('str').str.replace(r'v\d', '')
            return df

        merged_data = fix_pid_cols(merged_data)
        merged_data['p_id'].isna().value_counts()

        # ### which sub-department has the most transactions?
        merged_data.groupby(['sub_department_desc']).size().sort_values(ascending=False)

        # how many product types exist within knits and dresses?
        print('unique products in knits (p_id): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'W L/S KNITS']['p_id'].nunique()))
        print('unique products in knits (p_id_x): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'W L/S KNITS']['p_id_x'].nunique()))
        print('unique products in knits (p_id_y): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'W L/S KNITS']['p_id_y'].nunique()))
        print('unique products in knits (style): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'W L/S KNITS']['style'].nunique()))
        print('unique products in dresses (p_id): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'DRESSES']['p_id'].nunique()))
        print('unique products in dresses (p_id_x): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'DRESSES']['p_id_x'].nunique()))
        print('unique products in dresses (p_id_y): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'DRESSES']['p_id_y'].nunique()))
        print('unique products in dresses (style): ' + str(
            merged_data[merged_data['sub_department_desc'] == 'DRESSES']['style'].nunique()))

        merged_data[merged_data['sub_department_desc'] == 'W L/S KNITS'][
            ['p_id', 'p_id_x', 'p_id_y', 'style']].drop_duplicates().sort_values('style')

        # ### simplify colours
        def simplify_colours(df):
            '''
            Simplifies colours present in the 'color' column of dataframe.
                Parameters:
                    df (df): Dataframe in which to simplify colours.
                Returns:
                    df (df): A dataframe containing an extra column 'color_simple' that contains simplified colours.
            '''

            df['color_simple'] = df['color']

            contains_colour = ['Windsor Heather', 'Zoe Wash', 'Woodbury Strp Wh/Hydran/G',
                               'Woodburn Patchwork', 'Wooster/Alley', 'Woodley Plaid',
                               'York Pld/Oxford Pld Gld', 'Watercolor Print 1', 'Vintage Sailboat',
                               'Wiggins', 'Winona Wash', 'Vintage Port Multi', 'Williams Wash'
                , 'Yucatan', 'Rose', 'Gold', 'Wine', 'Navy', 'Royal', 'Wisteria', 'Whiskey', 'Coral',
                               'Lavender', 'Tan', 'Khaki', 'Camo', 'Taupe', 'Wildflower', 'Floral', 'Hibiscus',
                               'Silver',
                               'Pepper', 'Vicuna', 'Washed Forest', 'Mauve', 'Camel', 'Light Indigo', 'Whyskey',
                               'Windsor Heather Multi Str',
                               'Woodbridge Olive', 'Yuca Tan', 'Yanda Wash', 'White', 'Zebra', 'Blue', 'Cream', 'Navy',
                               'Grey', 'Orange',
                               'Green', 'Red', 'Rose', 'Mauve', 'Purple', 'Black', 'Pink', 'Brown', 'Yellow']

            replace_colour = ['Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other',
                              'Other',
                              'Other', 'Other', 'Other', 'Pink', 'Yellow', 'Purple', 'Blue', 'Blue', 'Purple', 'Brown',
                              'Red', 'Purple', 'Brown', 'Green',
                              'Green', 'Brown', 'Floral', 'Floral', 'Pink', 'Grey', 'Grey', 'Brown', 'Green', 'Purple',
                              'Brown', 'Blue', 'Brown', 'Windsor Heather',
                              'Green', 'Yucatan', 'White', 'White', 'Zebra', 'Blue', 'Cream', 'Navy', 'Grey', 'Orange',
                              'Green', 'Red', 'Rose', 'Mauve', 'Purple', 'Black',
                              'Pink', 'Brown', 'Yellow']

            for ii in range(len(df['color_simple'])):
                for jj in range(len(contains_colour)):
                    if contains_colour[jj] in df['color_simple'][ii]:
                        df['color_simple'][ii] = replace_colour[jj]
                        break
            return df

        merged_data = simplify_colours(merged_data)

        # ### select columns of interest, sub-category of interest and positive transactions
        def select_columns_category(df):
            '''
            Creates a new dataframe that contains columns to take forward to the next step, rows matching the 'W L/S KNITS' subdepartment,
            and rows deonoting positive transactions.
                Parameters:
                    df (df): Dataframe from which to extract columns and rows.
                Returns:
                    df (df): Dataframe containing columns and rows of interest.

            '''
            df = df[
                ['p_id', 'transaction_date', 'sub_department_desc', 'label_desc', 'color_simple', 'quantity', 'amount']]
            df = df[df['sub_department_desc'] == 'W L/S KNITS']
            df = df[df['amount'] > 0]
            return (df)

        merged_data_pruned_sd_knits = select_columns_category(merged_data)
        # merged_data_pruned_sd_knits.shape

        # ### write data to interim folder
        # merged_data_pruned_sd_knits.to_csv(Path(self.path_data / 'interim/transactions_sd_knits.csv'), index=False)

        self.merged_data_pruned_sd_knits = merged_data_pruned_sd_knits
        return merged_data_pruned_sd_knits

    def data_synthesis(self, knit_data):
        '''
        Resampling original data.
            Parameters:
                knit_data: Dataframe of merged transactions from data_process_cww.
            Returns:
                sample_manual: Dataframe of resampled data.
        '''

        # knit_data = pd.read_csv(Path(self.path_data / 'interim/transactions_sd_knits.csv'))
        # knit_data.info()

        # ### resample data using pandas sample
        def resample_data(df):
            '''
            Randomly resamples rows with replacement from input data, fraction of axis items to return is set to 80.
                Parameters:
                    df (df): Dataframe from which to resample rows.
                Returns:
                    df (df): Dataframe containing resampled rows.
            '''
            resampled_df = knit_data.sample(frac=80, replace=True, random_state=1)
            return resampled_df

        sample_manual = resample_data(knit_data)

        # ### compare original and resampled data
        print('knit data shape: ' + str(knit_data.shape))
        print('resampled data shape: ' + str(sample_manual.shape))
        print('knit data unique products: ' + str(knit_data['p_id'].nunique()))
        print('resampled data unique products: ' + str(sample_manual['p_id'].nunique()))

        knit_data['transaction_date'] = pd.to_datetime(knit_data['transaction_date'], infer_datetime_format=True)
        sample_manual['transaction_date'] = pd.to_datetime(knit_data['transaction_date'], infer_datetime_format=True)

        sns.relplot(
            data=knit_data.groupby(['transaction_date']).size().to_frame("count").reset_index(),
            x="transaction_date",
            y="count",
            aspect=3,
            kind="line",
            height=4
        )

        sns.relplot(
            data=sample_manual.groupby(['transaction_date']).size().to_frame("count").reset_index(),
            x="transaction_date",
            y="count",
            aspect=4,
            kind="line",
            height=3
        )

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        sns.histplot(knit_data["label_desc"].astype('category'), ax=ax[0]).set(title='knit_data')
        sns.histplot(sample_manual["label_desc"].astype('category'), ax=ax[1]).set(title='sample_manual')
        fig.show()

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        sns.histplot(knit_data["color_simple"].astype('category'), ax=ax[0]).set(title='knit_data')
        sns.histplot(sample_manual["color_simple"].astype('category'), ax=ax[1]).set(title='sample_manual')
        fig.show()

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        sns.histplot(knit_data["quantity"].astype('category'), ax=ax[0]).set(title='knit_data')
        sns.histplot(sample_manual["quantity"].astype('category'), ax=ax[1]).set(title='sample_manual')
        fig.show()

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        sns.histplot(knit_data["amount"], ax=ax[0]).set(title='knit_data')
        sns.histplot(sample_manual["amount"], ax=ax[1]).set(title='sample_manual')
        fig.show()

        # ### remap identifiable data
        def remap_identifiable_data(df):
            '''
            Remaps identifying information in the 'p_id', 'sub_department_desc', are 'label_desc' columns of the original data.
                Parameters:
                    df (df): Dataframe in which to remap identifying information.
                Returns:
                    df (df): Dataframe with identifying information remapped.
            '''
            p_id_array = df['p_id'].unique()
            dict_p_id = {}
            for i in range(len(p_id_array)):
                dict_p_id[p_id_array[i]] = "p_" + str(i + 1)

            dict_sub_dept = {'W L/S KNITS': 'KNITS'}

            label_desc_array = df['label_desc'].unique()
            dict_label_desc = {}
            for i in range(len(label_desc_array)):
                dict_label_desc[label_desc_array[i]] = "lab_" + str(i + 1)

            df.replace({"p_id": dict_p_id, "sub_department_desc": dict_sub_dept, "label_desc": dict_label_desc},
                       inplace=True)

            return df

        sample_manual = remap_identifiable_data(sample_manual)
        # sample_manual.head()

        # ### write resampled data to interim folder
        # sample_manual.to_csv(Path(self.path_data / 'interim/transactions_sd_knits_resampled.csv'), index=False)

        self.sample_manual = sample_manual
        return sample_manual

    def feature_add(self, knit_data):
        '''
        Additional feature engineering, including star rating, sentiment, and google trend.
            Parameters:
                knit_data: Dataframe of resampled data from data_synthesis.
            Returns:
                knit_data: Dataframe with additional feature engineering.
        '''

        # knit_data = pd.read_csv(Path(self.path_data / 'interim/transactions_sd_knits_resampled.csv'))
        knit_data['transaction_date'] = pd.to_datetime(knit_data['transaction_date'], infer_datetime_format=True)

        # ### engineer columns
        def engineer_columns(df):
            '''
            Attaches engineered columns including 'week_no', 'month', 'price' per week, 'price_comp_week' and 'transaction_date' for the first Sunday of a week.
                Parameters:
                    df (df): Dataframe to engineer and attach columns to.
                Returns:
                    df (df): Dataframe with engineered columns attached.
            '''

            # sort data
            df = df.sort_values(by=['p_id', 'transaction_date'])

            # add week no
            df['week_no'] = df['transaction_date'].dt.strftime('%U')

            # concatenate by week and sum quantity
            knit_data_week = df.groupby(['p_id', 'week_no', 'sub_department_desc', 'label_desc', 'color_simple'])[
                'quantity'].sum().to_frame('quantity').reset_index()

            # Calculate price as average of amount per week / quantity
            sum_amount_week = df.groupby(['p_id', 'week_no', 'sub_department_desc', 'label_desc', 'color_simple'])[
                'amount'].sum().to_frame('sum_amount').reset_index()
            knit_data_week = pd.merge(knit_data_week, sum_amount_week,
                                      on=['p_id', 'week_no', 'sub_department_desc', 'label_desc', 'color_simple'],
                                      how='left')
            knit_data_week['price'] = knit_data_week['sum_amount'] / knit_data_week['quantity']
            knit_data_week.drop(columns=['sum_amount'], inplace=True)
            df = knit_data_week

            # engineer price competition column per week
            price_mean_week = df.groupby(['week_no'])['price'].mean().to_frame("mean_price_week").reset_index()
            df = pd.merge(df, price_mean_week, on='week_no', how='left')
            df['price_comp_week'] = df['price'] / df['mean_price_week']
            df.drop(columns=['mean_price_week'], inplace=True)

            # add date for first sunday back in
            def find_sunday(week):
                ref = '2021-01-03'  # reference date corresponding to the 1st Sunday in 2021
                ref_object = dt.strptime(ref, '%Y-%m-%d')  # reference day converted in datetime object
                day_object = ref_object + timedelta(days=(week - 1) * 7)  # adding the number of days
                day = day_object.strftime('%Y-%m-%d')  # converting back to the desired format
                return day

            df['transaction_date'] = df['week_no'].apply(lambda x: find_sunday(int(x)))

            # add month column
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], infer_datetime_format=True)
            df['month'] = df['transaction_date'].dt.strftime('%b')

            df = df.sort_values(by=['p_id', 'transaction_date'])

            return df

        knit_data = engineer_columns(knit_data)

        # ### define function - star rating and sentiment
        def synthesise_star_sentiment(df, random_seed=123, star_dist=[0.03675, 0.06773, 0.12719, 0.23374, 0.53459]):
            '''
            Returns two synthetic features i) star_rating (1-5) randomly assigned based on the distribution profile
            of star ratings given to women's knitwear in the following dataset:
            https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews?datasetId=11827&sortBy=voteCount
            Future star rating could be assigned to a transaction based on the average rating of an item at the time of transaction.
            ii) review sentiment, -1 for negative, 0 for neutral and 1 for positive, assigned to transaction based on
            star rating. Distribution of assignment was based on educated guess. In future actual reviews could be analysed
            by Natural language processing to determine the average review sentiment of an item at the time of tranaction.

                    Parameters:
                        df (dataframe): Dataframe to append new features of star rating and review sentiment
                        random_seed (int): Random number (default is 123)
                        star_dist (list): List of floats which represent the distribution profile
                    Returns:
                        df (dataframe): Dataframe with appended columns 'star_rating' for assigned star rating and
                         'review' to capture review sentiment
            '''
            np.random.seed(random_seed)

            df['star_rating'] = np.nan

            df['star_rating'] = np.random.choice([1, 2, 3, 4, 5], size=df.shape[0], replace=True, p=star_dist)

            mask1 = df['star_rating'] >= 4
            mask2 = (df['star_rating'] < 4) & (df['star_rating'] >= 2)
            mask3 = df['star_rating'] < 2

            df.loc[mask1, 'review'] = 1
            df.loc[mask2, 'review'] = np.random.choice([-1, 0, 1], replace=True, p=[.2, .65, .15])
            df.loc[mask3, 'review'] = np.random.choice([-1, 0, 1], replace=True, p=[.9, .08, .02])

            df['review'] = df['review'].astype(str)

            return df

        # star_dist_sales = [0, 0.07, 0.14, 0.24, 0.55]
        # star_dist_returns = [0.18, 0.2, 0.26, 0.36, 0])

        # ### synthesise data
        knit_data = synthesise_star_sentiment(df=knit_data)
        # knit_data['star_rating'].hist()
        # knit_data['review'].hist()

        search_list = ['Black Knit', 'White Knit', 'Zebra Knit', 'Blue Knit', 'Green Knit',
                       'Pink Knit', 'Yellow Knit', 'Cream Knit', 'Brown Knit']

        search_start = '2020-12-27 '
        search_end = '2021-12-31'

        # ### Functions to download google trends data and convert to interest rate of change
        def get_google_trends(search_list, search_start, search_end):
            '''
            Returns new dataframe of google trend's weekly interest over time of a search term in search list.
            Values are relative to the highest search volume of that term in the defined time period in a given
            georgraphical region (here GB). A value of 100 is the peak popularity for the term, 50 means that
            the term is half as popular. A score of 0 means that there was not enough data for this term. Designed
            for function to be run for each new google_trend feature to be added to the dataframe, which is paired
            to another feature in the dataframe, with a search list containing the categorical column options.

                    Parameters:
                        search_list (list of str): Terms to be searched - corresponds to different categorical
                        search_start (str): date in 'YYYY-MM-DD' to avoid NaN downstream date should be two weeks before
                                            start of corresponding transactions dataframe
                        search_end (str): date in 'YYYY-MM-DD'

                    Returns:
                        dataframe of all search terms with google ternds relative interest over specified time period
            '''

            pytrend = TrendReq(hl='en-UK', tz=0)
            trends_dict = {}
            df_trends = pd.DataFrame()

            for term in search_list:
                pytrend.build_payload(
                    kw_list=[term],
                    cat=0,
                    timeframe=(search_start + search_end))
                trends_dict[term] = pytrend.interest_over_time()

            df_trends = pd.concat([trends_dict[key] for key in search_list], join='inner', axis=1)
            return df_trends.drop(labels=['isPartial'], axis='columns')

        # Google trends interest relative to the first week of 2021
        def calculate_google_trend_term_relative_to_week_1(df):
            '''
            Returns google trend dataframe with interest over time relative to the interest in first week in 2021

                    Parameters:
                        df (dataframe): google trends dataframe

                    Returns:
                        df (dataframe): google trends dataframe with values relative to week 1 2021

            '''

            return df.div(df.iloc[1])  # Week1 2021 is not first row of this dataframe hence position [1]

        # Redundant for linear model but could be useful for ARIMA?

        # Function to compare current google trend information to the 3 week moving average as a rate of change
        # Negative values imply term is searched less than the 3 week moving average, Positive values, more than.
        def calculate_rate_of_change_of_google_trend_term(df):
            '''
            Function to compare current google trend information to the 3 week moving average as a rate of change
            Negative values imply term is searched less than the 3 week moving average, Positive values, more than.
            NB: Not used in the end for the project - maybe useful for an ARIMA model?

                    Parameters:
                        df (dataframe): google trends dataframe

                    Returns:
                        df_diff_MA (dataframe): google trends dataframe with interest values relative to 3 week moving average
            '''

            moving_dict = {}

            for col in df.columns:
                moving_dict[col] = df[col].rolling(3, min_periods=3).mean()
            df_moving_average = pd.concat([moving_dict[col] for col in df.columns], join='inner', axis=1)

            df_diff_MA = df.subtract(df_moving_average, axis=1)

            return df_diff_MA.div(df_moving_average, axis=1)

        # Function to expand out trends from week to days
        def make_trends_daily(df, search_start, search_end):
            '''
            Returns expanded google trends dataframe on daily rather than weekly basis
            (forward fill of sunday value)

                    Parameters:
                        df (dataframe): google trends dataframe with values relative to week 1 2021

                    Returns:
                        df (dataframe): daily google trends dataframe with values relative to week 1 2021

            '''

            date_range = pd.date_range(start=search_start, end=search_end)
            return df.reindex(date_range).fillna(method='ffill')

        # Google trend feature for colour of knit
        trend_df = get_google_trends(search_list, search_start, search_end)
        colour_trend = calculate_google_trend_term_relative_to_week_1(trend_df)
        colour_trend_daily = make_trends_daily(colour_trend, search_start, search_end)

        # Google trend feature relating to style of 'knits'
        style_knits = get_google_trends(['Knits'], search_start, search_end)
        trend_knits = calculate_google_trend_term_relative_to_week_1(style_knits)
        knits_trend_daily = make_trends_daily(trend_knits, search_start, search_end)
        knits_trend_daily.reset_index(inplace=True)

        # ### Convert to feature on the transactions dataframe
        # knits_transactions = pd.read_csv("../data/interim/transactions_sd_knits_resampled_synth.csv") ### aminah
        # knits_transactions['transaction_date'] = knits_transactions['transaction_date'].apply(pd.to_datetime) ### aminah

        # ### Function to add sub department knits google trend
        # TOFIX: This will not work if more than one sub department is presented to be searched for in the google trends

        # Function to append the subdepartment 'knit' google trend rate of change based on the date.
        def append_google_trends_sub_depart_feature(df_transactions, df_google_trends):
            '''
            Returns column appended to transaction data frame with the google trend interest of the subdepartment
            term ie 'knits' based on the date of transaction. NB this for use of appending google trends
            data where there is only one search term otherwise use function append_google_trends_colour_feature.

                    Parameters:
                        df_transactions (dataframe): dataframe with tranactions and physical attributes of items
                        df_google_trends (dataframe): daily google trends dataframe with values relative to week 1 2021


                    Returns:
                        df_trans_gt_knit (dataframe): transactions dataframe with appended 'google_trend_knit' variable

            '''

            df_trans_gt_knit = df_transactions.merge(df_google_trends,
                                                     left_on='transaction_date',
                                                     right_on='index',
                                                     how='left')

            df_trans_gt_knit.drop(columns=['index'], inplace=True)
            df_trans_gt_knit.rename(columns={'Knits': 'google_trends_knit'}, inplace=True)

            return df_trans_gt_knit

        knits_transactions_gt_knits = append_google_trends_sub_depart_feature(knit_data, knits_trend_daily)  ### aminah

        # Function extracts the correct google_trends column based on the colour of the product in the transaction
        # and passes in the google trend rate of change based on the date.
        def append_google_trends_colour_feature(df_transactions, df_google_trends):
            '''
            Returns transaction dataframe with the appropriate google_trends value based on the colour in 'color_simple'
            and the transaction date for each transaction.

                    Parameters:
                        df_transactions (dataframe): dataframe with tranactions and physical attributes of items
                        df_google_trends (dataframe): daily google trends dataframe with values relative to week 1 2021

                    Returns:
                        df_transactions (dataframe): transactions dataframe with appended 'google_trend_colour' variable

            '''

            df_transactions['google_trends_colour'] = np.nan

            df_google_trends.columns = df_google_trends.columns.str.replace(' Knit', '')

            for i, row in df_transactions.iterrows():

                # get date and color in transactions
                transaction_date = f"{row['transaction_date']}"
                x = f"{row['color_simple']}"

                if x != 'Other':
                    # get correct colour column
                    gt_colour = df_google_trends[df_google_trends.columns[df_google_trends.columns.isin([x])]]
                    gt_colour.reset_index(inplace=True)

                    mask = gt_colour['index'] == transaction_date

                    gt = gt_colour.loc[mask, x]

                    gt = gt.values

                    df_transactions['google_trends_colour'].iloc[i] = gt
                else:
                    df_transactions['google_trends_colour'].iloc[i] = 0  # No change in other trend

            return df_transactions

        append_google_trends_colour_feature(knits_transactions_gt_knits, colour_trend_daily)

        knit_data = knits_transactions_gt_knits

        # ### drop knit column redundant
        knit_data.drop(columns=['sub_department_desc'], inplace=True)

        # ### rearrange columns
        knit_data = knit_data[['p_id', 'transaction_date', 'week_no', 'month', 'label_desc', 'color_simple', 'quantity',
                               'price', 'price_comp_week', 'star_rating', 'review', 'google_trends_knit',
                               'google_trends_colour']]

        # ### Sort on date
        knit_data = knit_data.sort_values(by=['transaction_date'])

        # ### write to interim folder
        knit_data.to_csv(Path(self.path_data / 'interim/transactions_sd_knits_resampled_engin_synth_gt.csv'), index=False)
        # knit_data.info()

        self.knit_data = knit_data
        return knit_data

    def lp_mip_solver(self, demands, prices, sum_prices):
        '''
        Returns the optimum price to attribute to each item and the corresponding objective function (revenue). Optimisation solver giving the same result as MIP solver by using a combination of MIP and LP solvers (see Analytics for an Online Retailer: Demand
        Forecasting and Price Optimization, Ferreira et al.)

                Parameters:
                        demands (ndarray): matrix of dimension (n,k,j) with n the # of items,  the # of price possibilities, and j # the number of sum of prices considered
                        prices (ndarry): 1D-array containing the 'k' possible prices to attribute to the items
                        sum_prices (ndarray): 1D-array containing the sum of prices considered (vector typically ranging from n*min(prices) to n*max(prices))

                Returns:
                        optimal_prices (ndarray): 1D-array containing the prices to attribute to each item
                        revenue_prediction (float): revenue corresponding to the prices attributed
        '''

        def loop_k(demands, prices, sum_prices, A):
            # loop calling LP solvers
            # function where the loop over the different values of the sum of the prices: sum_prices[aa] is executed
           
            n, k, num_loops = np.shape(demands)

            

            # initialisation
            objective_loop = np.zeros(len(sum_prices))  # best objective function in each solution
            LBk = np.zeros(len(sum_prices))

            for aa in range(num_loops):
                demands_submatrix = demands[:, :, aa]

                # r is the vector giving the different price * demand combinations
                # It is used to define the cost function to maximise: max_x( tranpose(r) * x )
                r = np.multiply(np.tile([prices], n), np.array(demands_submatrix).reshape(1, k * n))

                b = [np.append(np.ones(n), sum_prices[aa])]  # constraint vector: [1....1 sum_prices[aa]]

                lp_sol = linprog(-r.flatten(), A_eq=A, b_eq=b)  # calling and initialiasing the model 

                if lp_sol.status == 0:
                    objective_loop[aa] = -lp_sol.fun
                    LBk[aa] = -lp_sol.fun - np.max(
                        np.max(prices * demands_submatrix, axis=1) - np.min(prices * demands_submatrix, axis=1))

            return objective_loop, LBk

        n, k, num_loops = np.shape(
            demands)  # n corresponds to the number of products and k to the number of prices considered in the optimisation problem

        # sanity check
        assert num_loops == len(sum_prices), 'the demands matrix last dimension is different from len(sum_prices)'
        assert k == len(prices), 'the demands matrix middle dimension is different from len(prices)'

        # initialising empty variables
        optimum_solution = np.zeros(n * k)  # optimum solution
        revenue_prediction = 0  # optimum revenue

        # Constraints are recast out of the shape A*x = b.
        # Two types of constraints are considered. The sum of the prices of a single item must be equal to 1 (for a binary variable this means that an item has only a single price!)
        A = np.array([[
            1 if j >= k * (i) and j < k * (i + 1) else 0
            for j in range(k * n)
        ] for i in range(n)])

        # The second set of constraints is defined and added to A here: The sum of the prices must be equal to sum_prices[aa]
        A = np.append(A, np.tile([prices], n), axis=0)

        objective_loop, LBk = loop_k(demands, prices, sum_prices, A)

        # step 2 (LP bound Algorithm) in `Analytics for an Online Retailer: Demand Forecasting and Price Optimization`
        sum_prices_sorted = sum_prices[np.argsort(objective_loop)[::-1]]
        objective_sorted = objective_loop[np.argsort(objective_loop)[::-1]]
        demands_sorted = demands[:, :, np.argsort(objective_loop)[::-1]]
        LBk_sorted = LBk[np.argsort(objective_loop)[::-1]]

        # step 3 (LP bound Algorithm)
        k_hat = np.argmax(LBk_sorted)
        LB = LBk_sorted[k_hat]

        ll = 0
        flag = True
        while flag == True:  # looping over the MIP problems while flag == True

            demands_submatrix = demands_sorted[:, :, ll]
            r = np.multiply(np.tile([prices], n), np.array(demands_submatrix).reshape(1, k * n)).flatten()
            b = [np.append(np.ones(n), sum_prices_sorted[ll])]  # constraint vector: [1....1 sum_prices[ll]]
            m = Model()  # calling the model object and initiation

            x = [m.add_var(var_type=BINARY) for i in
                 range(k * n)]  # defining the different variables: n*k variables and defining them as binary

            m.objective = maximize(
                xsum(r[i] * x[i] for i in range(k * n)))  # objective function defined as r*x to be maximised

            for j in range(n + 1):  # adding the different constraints to the problem Ax = b
                m += xsum(A[j, i] * x[i] for i in range(n * k)) == b[0][j]

            status = m.optimize()  # calling the solver

            if status == OptimizationStatus.OPTIMAL and m.objective_value > LB:
                k_hat = ll
                LB = m.objective_value
                revenue_prediction = m.objective_value  # then we want this to be our objective
                optimum_solution = np.array([x[aa].x for aa in range(n * k)])  # recording the solution

            if ll == num_loops - 1:
                flag = False
            elif status == OptimizationStatus.OPTIMAL and LB >= objective_sorted[ll + 1]:
                flag = False
            else:
                ll += 1

        optimal_prices = np.matmul(optimum_solution.reshape(n, k),
                                   prices)  # returning the vector of optimum prices for each item

        return optimal_prices, revenue_prediction

    def predict_demand(self, knit_data):
        '''
        Demand prediction model.
            Parameters:
                knit_data (df): Dataframe of additional feature engineering from feature_add.
            Returns:
                demand_matrix (ndarray): Predicted demands matrix of dimension (n,k,j) with n the # of items, k the # of price possibilities,
                                         and j # the number of sum of prices considered
                prices (ndarray): 1D-array containing the 'k' possible prices to attribute to the items.
                sum_prices (ndarray): 1D-array containing the sum of prices considered.
        '''

        # ### Build prediction model *Can be modified*
        knit_data = pd.read_csv("../data/interim/transactions_sd_knits_resampled_engin_synth_gt.csv")

        def create_unseen_data(df):
            '''
            Creates a copy of 'unseen data', that is not used to train the model. 'unseen data' is used to compare real prices to predicted optimal prices.
                Parameters:
                    df (df): Dataframe from which to extract 'unseen data'.
                Returns:
                    df (df): Dataframe containing unseen data.
            '''
            unseen_data = df[df['transaction_date'] >= '2021-10-3']
            return unseen_data

        unseen_data = create_unseen_data(knit_data)

        # DO NOT ADD TO CLASS
        def prepare_data(df):
            knit_data['transaction_date'] = pd.to_datetime(knit_data['transaction_date'], infer_datetime_format=True)
            knit_data['week_no'] = knit_data['week_no'].astype('object')
            knit_data['review'] = knit_data['review'].astype('object')
            knit_data.drop(columns=['month'], inplace=True)
            knit_data.drop(columns=['p_id'], inplace=True)
            return (df)

        knit_data = prepare_data(knit_data)

        def one_hot_encode_categorical_aa(df):
            '''
            One hot encodes categorical variables.
                Parameters:
                    df (df): Dataframe to one hot encode.
                Returns:
                    df_encoded (df): Dataframe including one hot encoded
                    ohe_dropped_cols (list): List of one hot encoded columns that were dropped to get k-1 columns.
            '''
            df_encoded = pd.get_dummies(df)
            # drop columns to get k-1 columns for
            # ohe_dropped_cols = ['week_no_02', 'label_desc_lab_1', 'color_simple_Other', 'review_0.0']
            ohe_dropped_cols = ['week_no_2', 'label_desc_lab_1', 'color_simple_Other', 'review_0.0']
            df_encoded.drop(columns=ohe_dropped_cols,
                            axis=1,
                            inplace=True)
            return df_encoded, ohe_dropped_cols

        knit_data, ohe_dropped_cols = one_hot_encode_categorical_aa(knit_data)

        # DO NOT ADD TO CLASS
        def log_price_quantity(df):
            # take log of price and quantity, drop original columns
            df['price_log'] = np.log(df['price'] + 1)
            df['quantity_log'] = np.log(df['quantity'] + 1)
            df.drop(columns=['price'], inplace=True)
            df.drop(columns=['quantity'], inplace=True)

            return df

        knit_data = log_price_quantity(knit_data)

        def build_prediction_model(df):
            '''
            Builds and returns a centralised random forest model based on best parameters, trained on dates that are treated as 'historic',
            Also returns training data.
                Parameters:
                    df (df): Dataframe from which to extract 'historic' data.
                Returns:
                    RF_cen_model (model): RandomForestRegressor.fit() object.
                    X_train (df): Dataframe used to train random forest model.
                    y_train (df): Dataframe used to train random forest model.

            '''
            df_train = df[df['transaction_date'] < '2021-10-3']
            y_train = df_train['quantity_log']
            X_train = df_train.drop(['quantity_log', 'transaction_date'], axis=1)

            RF_cen_model = RandomForestRegressor(max_features=50,
                                                 max_depth=6,
                                                 n_estimators=820,
                                                 random_state=0,
                                                 min_samples_split=2,
                                                 min_samples_leaf=2,
                                                 criterion='squared_error',
                                                 bootstrap=False
                                                 ).fit(X_train, y_train)

            return RF_cen_model, X_train, y_train

        RF_cen_model, X_train_historic, y_train_historic = build_prediction_model(knit_data)

        # ### apply model to unseen data
        def build_X_unseen(X_historic, week_no, label_desc, color_simple, price, relative_price,
                           ohe_dropped_cols=ohe_dropped_cols):
            '''
            Builds dataframe containing one row, which is used to predict demand for 'unseen'/future items.
                Parameters:
                    X_historic (df): Dataframe used to train random forest model.
                    week_no (int): Week in which 'unseen'/future item is to be sold. For testing pipeline, values between 44 - 52 are sensible.
                    label_desc (object): Label to which 'unseen'/future item belongs. Takes the following values: 'lab_1', 'lab_2', 'lab_3', 'lab_4'.
                    color_simple (object): Colour of 'unseen'/future item. Takes the following values: 'White', 'Pink', 'Black', 'Other', 'Blue',
                                            'Zebra', 'Yellow', 'Green', 'Brown', 'Cream'.
                    price (float): Suggested price of 'unseen'/future item.
                    relative_price (float): Suggested relative price of 'unseen'/future item.
                    ohe_dropped_cols (list): List of one hot encoded columns that were dropped in data preparation.
                Returns:
                    X_unseen (df): Dataframe used to predict demand for an 'unseen' item.
            '''
            # TODO throw error if variables other than those expected are shown e.g. a new label, new colour, unknown week

            # create output df with same colnames as training data, add in one hot encoded columns that were dropped in previous steps
            columns_all = list(X_historic.columns)
            columns_all += ohe_dropped_cols
            X_unseen = pd.DataFrame(columns=columns_all)
            row_dict = {'price_log': np.log(price + 1), 'price_comp_week': relative_price}
            X_unseen = X_unseen.append(row_dict, ignore_index=True)

            # fill in one hot encoded columns
            week_no_match = 'week_no_0' + str(week_no)
            X_unseen[week_no_match] = 1

            label_desc_match = 'label_desc_' + label_desc
            X_unseen[label_desc_match] = 1

            color_simple_match = 'color_simple_' + color_simple
            X_unseen[color_simple_match] = 1

            # add label and colour columns back into historic data
            X_historic['label_desc_lab_1'] = np.where(
                X_historic[['label_desc_lab_2', 'label_desc_lab_3', 'label_desc_lab_4']].sum(axis=1) == 0, 1, 0)
            X_historic['color_simple_Other'] = np.where(
                X_historic[['color_simple_Black', 'color_simple_Blue', 'color_simple_Brown', 'color_simple_Cream',
                            'color_simple_Green', 'color_simple_Pink', 'color_simple_White', 'color_simple_Yellow',
                            'color_simple_Zebra']].sum(axis=1) == 0, 1, 0)

            # for rating, google trend, and review field, take the median value from historic data based on label and item colour as this information would not be available for
            # predicting demand live
            # TODO if historic data for 1+ years is available, match based on week too - this should give better predictive demand prediction
            if X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)].shape[0] > 0:
                X_unseen['star_rating'] = \
                X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)][
                    'star_rating'].median()
                X_unseen['google_trends_knit'] = \
                X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)][
                    'google_trends_knit'].median()
                X_unseen['google_trends_colour'] = \
                X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)][
                    'google_trends_colour'].median()
                neg_rev_av = X_unseen['review_-1.0'] = \
                X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)][
                    'review_-1.0'].median()
                pos_rev_av = X_unseen['review_1.0'] = \
                X_historic[(X_historic[label_desc_match] == 1) & (X_historic[color_simple_match] == 1)][
                    'review_1.0'].median()
                if neg_rev_av > pos_rev_av:
                    X_unseen['review_-1.0'] = 1
                    X_unseen['review_1.0'] = 0
                else:
                    X_unseen['review_-1.0'] = 0
                    X_unseen['review_1.0'] = 1

            # if a label and colour combination hasn't been seen before take the median all historic data
            # this would be more meaningful if historic data for 1+ years was available and could be matched for week
            else:
                X_unseen['star_rating'] = X_historic['star_rating'].median()
                X_unseen['google_trends_knit'] = X_historic['google_trends_knit'].median()
                X_unseen['google_trends_colour'] = X_historic['google_trends_colour'].median()
                neg_rev_av = X_unseen['review_-1.0'] = X_historic['review_-1.0'].median()
                pos_rev_av = X_unseen['review_1.0'] = X_historic['review_1.0'].median()

                if neg_rev_av > pos_rev_av:
                    X_unseen['review_-1.0'] = 1
                    X_unseen['review_1.0'] = 0

                else:
                    X_unseen['review_-1.0'] = 0
                    X_unseen['review_1.0'] = 1

            # fill remaining NAs with 0
            X_unseen = X_unseen.fillna(0)

            # drop one hot encoded that were added in earlier step
            X_unseen.drop(columns=ohe_dropped_cols, inplace=True)

            # drop columns that were added onto historic data
            X_historic.drop(columns=['label_desc_lab_1', 'color_simple_Other'], inplace=True)

            # order columns correctly
            X_unseen = X_unseen[X_historic.columns]

            return X_unseen

        def predict_demand(X_unseen):
            '''
            Predicts demand for input data.
                Parameters:
                    X_unseen: Dataframe containing 1 row.
                Returns:
                    prediction: Prediction for 'unseen'/future item as an interpretable value.
            '''
            prediction = RF_cen_model.predict(X_unseen)
            prediction = np.exp(prediction) - 1
            prediction = np.round(prediction)
            prediction = prediction[0]

            return prediction

        # ### select competing products
        unseen_data.groupby(['week_no']).size().sort_values(ascending=False)

        # Test/unseen data is from week 44 - 52
        def select_competing_products(unseen_data, n_products, week_no, pc_lower_price_bound, pc_upper_price_bound,
                                      random_state):
            '''
            Randomly selects competing products and related features to test demand prediction and optimisation step.
                Parameters:
                    unseen_data (df): Dataframe from which to randomly select competing items. This should be the 'unseen'/future data.
                    n_products (int): Number of competing items to select. This should not be greater than the number of items that were actually sold for a particular week.
                    week_no (int): Week number to select competing items from. For testing pipeline, values between 44 - 52 are sensible.
                    pc_lower_price_bound (int): Percentage value by which to lower the price of an item for testing price optimisation.
                    pc_upper_price_bound (int): Percentage value by which to increase the price of an item for testing price optimisation.
                    random_state (int): Random state for reproducibility.
                Returns:
                    competing_items_dict (dict): A dictionary of competing items, including item features such as week of same, label, colour, lower price bound and upper
                                                 price bound.
            '''
            unseen_data = unseen_data[unseen_data['week_no'] == week_no]
            unseen_sample = unseen_data.sample(n=n_products, replace=False, random_state=random_state)
            unseen_sample_details = unseen_sample[['week_no', 'label_desc', 'color_simple', 'price']].reset_index(
                drop=True)

            competing_items_dict = {}
            for i in range(len(unseen_sample_details)):
                prod_name = 'unseen_' + str(i + 1)
                price = unseen_sample_details.iloc[i][3]
                lpb = round(price - (price * (pc_lower_price_bound / 100)), 2)
                upb = round(price + (price * (pc_upper_price_bound / 100)), 2)
                array = [unseen_sample_details.iloc[i][0], unseen_sample_details.iloc[i][1],
                         unseen_sample_details.iloc[i][2], lpb, upb]
                competing_items_dict[prod_name] = array

            return competing_items_dict

        def calc_revenue_get_prices(unseen_data, n_products, week_no, random_state):
            '''
            Calculates and returns total revenue and returns prices for competing products, input should match those used in select_competing_products().
                Parameters:
                    unseen_data (df): Dataframe from which to randomly select competing items. This should be the 'unseen'/future data.
                    n_products (int): Number of competing items to select. This should not be greater than the number of items that were actually sold for a particular week.
                    week_no (int): Week number to select competing items from. For testing pipeline, values between 44 - 52 are sensible.
                    random_state (int): Random state for reproducibility.
                Returns:
                    total_revenue (float): Total revenue for competing items in specified week.
                    actual_prices (list): Prices for competing items in specified week.

            '''
            unseen_data = unseen_data[unseen_data['week_no'] == week_no]
            unseen_sample = unseen_data.sample(n=n_products, replace=False, random_state=random_state)
            unseen_sample['revenue'] = unseen_sample['price'] * unseen_sample['quantity']
            total_revenue = round(unseen_sample['revenue'].sum(), 2)
            actual_prices = list(round(unseen_sample['price'], 2))

            return total_revenue, actual_prices

        competing_items_dict = select_competing_products(unseen_data=unseen_data, n_products=8, week_no=48,
                                                         pc_lower_price_bound=10, pc_upper_price_bound=20,
                                                         random_state=0)
        self.total_revenue, self.actual_prices = calc_revenue_get_prices(unseen_data=unseen_data, n_products=8, week_no=48,
                                                               random_state=0)

        competing_items_dict

        # ### Build demand matrix
        def build_demand_matrix(min_price, max_price, increment, competing_items_dict, X_historic):
            '''
            Builds and returns array of prices, sum_prices and 3D matrix of demand predictions to be input into the price optimisation step.
                Parameters:
                    min_price (int): Minimum price to consider for items when predicitng their demand.
                    max_price (int): Maximum price to consider for items when predicitng their demand.
                    increment (int): Increment by which to increase searched prices for predicting demand.
                    competing_items_dict (dict): Dictionary of competing items, including item features such as week of same, label, colour, lower price bound and upper
                                                 price bound.
                    X_historic (df): Dataframe used to train random forest model.
                Returns:
                    demand_matrix (ndarray): Predicted demands matrix of dimension (n,k,j) with n the # of items, k the # of price possibilities,
                                             and j # the number of sum of prices considered
                    prices (ndarray): 1D-array containing the 'k' possible prices to attribute to the items.
                    sum_prices (ndarray): 1D-array containing the sum of prices considered.
            '''

            competing_items = len(competing_items_dict)
            competing_items_keys = list(competing_items_dict.keys())
            prices = list(range(min_price, max_price + increment, increment))
            sum_prices = np.arange(min_price * competing_items, max_price * competing_items + increment, increment)

            demand_matrix = np.zeros((competing_items, len(prices), len(sum_prices)))

            for nn in range(competing_items):
                week_no = competing_items_dict[competing_items_keys[nn]][0]
                label_desc = competing_items_dict[competing_items_keys[nn]][1]
                color_simple = competing_items_dict[competing_items_keys[nn]][2]
                lpb = competing_items_dict[competing_items_keys[nn]][3]
                upb = competing_items_dict[competing_items_keys[nn]][4]

                for j2, jj in enumerate(prices):

                    for k2, kk in enumerate(sum_prices):

                        if (jj >= lpb) & (jj <= upb):
                            relative_price = jj / (kk / competing_items)

                            # TODO this can be sped up
                            X_unseen = build_X_unseen(X_historic=X_historic,
                                                      week_no=week_no,
                                                      label_desc=label_desc,
                                                      color_simple=color_simple,
                                                      price=jj,
                                                      relative_price=relative_price
                                                      )

                            demand_matrix[nn, j2, k2] = predict_demand(X_unseen)

                        else:
                            demand_matrix[nn, j2, k2] = 0

            return demand_matrix, np.array(prices), sum_prices

        demand_matrix, prices, sum_prices = build_demand_matrix(min_price=20, max_price=370, increment=100,
                                                                competing_items_dict=competing_items_dict,
                                                                X_historic=X_train_historic)
        print(demand_matrix)

        self.demand_matrix, self.prices, self.sum_prices = demand_matrix, prices, sum_prices
        return demand_matrix, prices, sum_prices
