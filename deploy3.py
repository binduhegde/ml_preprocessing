from drop_columns import DropColumns
from feature_scaling import FeatureScaling
from encoding import Encoding
from missing_values import Imputation
import streamlit as st
import pandas as pd
import numpy as np
np.bool = np.bool_
np.object = object


class StreamlitApp:
    def __init__(self) -> None:
        st.title('Machine Learning Preprocessing')
        self.df_file = st.file_uploader('Upload the Dataframe', type=['csv'])

        if self.df_file is not None:
            self.df = pd.read_csv(self.df_file)
            self.get_options()

    def get_options(self):
        self.handle_null_values()
        self.encode_categorical_data()
        self.feature_scaling()
        self.drop_unnecessary_cols()
        if st.button('review my actions'):
            self.review_actions()

        # st.markdown('---')  # Add a horizontal line
        # if st.button('Download the preprocessed file'):
        #     self.submit_clicked()

    def handle_null_values(self):
        st.subheader('Step 1: Handling Null Values')
        st.text('These are the description of the columns with missing values')
        # Get columns with at least one missing value
        columns_with_na = [
            col for col in self.df.columns if self.df[col].isnull().any()]

        # Calculate number of missing values and data types for these columns
        na_info = {
            'Column': [],
            'Missing Values': [],
            'Data Type': [],
            'Mean': [],
            'Median': [],
            'Mode': []
        }
        for col in columns_with_na:
            na_info['Column'].append(col)
            na_info['Missing Values'].append(self.df[col].isnull().sum())
            na_info['Data Type'].append(self.df[col].dtype)
            if self.df[col].dtype != 'object':
                na_info['Mean'].append(self.df[col].mean())
                na_info['Median'].append(self.df[col].median())
            else:
                na_info['Mean'].append(np.nan)
                na_info['Median'].append(np.nan)
            na_info['Mode'].append(self.df[col].mode()[0])

        # Display the information
        na_info_df = pd.DataFrame(na_info)
        st.dataframe(na_info_df)

        self.remove_cols = st.multiselect(
            'I want to remove rows of these columns with missing values', columns_with_na)

        num_na_cols = [
            col for col in columns_with_na if self.df[col].dtype != 'object']
        self.fill_mean_cols = st.multiselect(
            'I want to fill the missing values in these columns with the mean', num_na_cols)

        self.fill_median_cols = st.multiselect(
            'I want to fill the missing values in these columns with the median', num_na_cols)

        self.fill_mode_cols = st.multiselect(
            'I want to fill the missing values in these columns with the mode', columns_with_na)

    def encode_categorical_data(self):
        st.subheader('Step 2: Encoding Categorical Data')
        st.text('These are the categorical feautures')
        cat_cols = self.df.select_dtypes(include='object').columns

        cat_info = {
            'Column': [],
            'Unique items': []
        }
        for col in cat_cols:
            cat_info['Column'].append(col)
            cat_info['Unique items'].append(self.df[col].nunique())
        cat_info_df = pd.DataFrame(cat_info)
        st.dataframe(cat_info_df)

        self.onehot_encoding_cols = st.multiselect(
            'I want to one-hot encode these columns', cat_cols)

        self.label_encoding_cols = st.multiselect(
            'I want to label-encode these columns', cat_cols)

    def feature_scaling(self):
        st.subheader('Step 3: Feature Scaling')
        st.text('These are the numerical columns')
        num_cols = self.df.select_dtypes(exclude='object').columns

        st.dataframe(self.df[num_cols].describe())
        self.minmax_scaler_cols = st.multiselect(
            'I want to Min-Max scale these columns', num_cols)

        self.std_scaler_cols = st.multiselect(
            'I want to Standard scale these columns', num_cols)

    def review_actions(self):
        self.set_user_pref()
        st.subheader('Step 5: Review the actions')
        if len(self.final_remove_cols) != 0:
            st.text('The following columns will be removed')
            st.text(' '.join(self.final_remove_cols))
        if len(self.final_fill_mean_cols) != 0:
            st.text(
                'The missing values in the following column will be filled with the mean')
            st.text(' '.join(self.final_fill_mean_cols))
        if len(self.final_fill_median_cols) != 0:
            st.text(
                'The missing values in the following column will be filled with the median')
            st.text(' '.join(self.final_fill_median_cols))
        if len(self.final_fill_mode_cols) != 0:
            st.text(
                'The missing values in the following column will be filled with the mode')
            st.text(' '.join(self.final_fill_mode_cols))

        if len(self.final_onehot_encoding_cols) != 0:
            st.text('The following columns will be one-hot encoded')
            st.text(' '.join(self.final_onehot_encoding_cols))
        if len(self.final_label_encoding_cols) != 0:
            st.text('The following columns will be label encoded')
            st.text(' '.join(self.final_label_encoding_cols))

        if len(self.final_cols_to_drop) != 0:
            st.text('The following columns will be dropped')
            st.text(' '.join(self.final_cols_to_drop))
        self.submit_clicked()

    def drop_unnecessary_cols(self):
        st.subheader('Step 4: Drop unnecessary columns')
        self.cols_to_drop = st.multiselect(
            'I want to drop these columns', self.df.columns)

    def submit_clicked(self):
        self.set_user_pref()
        self.preprocess()

        csv = self.result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Preprocessed Data",
            data=csv,
            file_name='preprocessed_data.csv',
            mime='text/csv',)
        # self.review_actions()
        # st.subheader('The following actions will be taken')

    def set_user_pref(self):
        def exclude_columns(selected_cols, *exclusions):
            """Helper function to exclude columns from selected_cols based on exclusions"""
            excluded_cols = set().union(*exclusions)
            return [col for col in selected_cols if col not in excluded_cols]

        # Final lists based on user selections
        self.final_remove_cols = self.remove_cols[:]

        self.final_fill_mean_cols = exclude_columns(
            self.fill_mean_cols, self.final_remove_cols)

        self.final_fill_median_cols = exclude_columns(
            self.fill_median_cols, self.final_remove_cols, self.final_fill_mean_cols)

        self.final_fill_mode_cols = exclude_columns(
            self.fill_mode_cols, self.final_remove_cols, self.final_fill_mean_cols, self.final_fill_median_cols)

        self.final_onehot_encoding_cols = self.onehot_encoding_cols[:]

        self.final_label_encoding_cols = exclude_columns(
            self.label_encoding_cols, self.final_onehot_encoding_cols)

        self.final_minmax_scaler_cols = self.minmax_scaler_cols[:]

        self.final_std_scaler_cols = exclude_columns(
            self.std_scaler_cols, self.final_minmax_scaler_cols)

        self.final_cols_to_drop = self.cols_to_drop[:]

    def preprocess(self):
        imputation = Imputation(self.df)
        imputation.remove_null(self.final_remove_cols)
        imputation.fill_mean(self.final_fill_mean_cols)
        imputation.fill_median(self.final_fill_median_cols)
        imputation.fill_mode(self.final_fill_mode_cols)
        after_imputation = imputation.data

        encode_categorical = Encoding(after_imputation)
        encode_categorical.onehot_encoding(self.final_onehot_encoding_cols)
        encode_categorical.label_encoding(self.final_label_encoding_cols)
        after_encoding = encode_categorical.data

        scaling = FeatureScaling(after_encoding)
        scaling.standard_scaling(self.final_std_scaler_cols)
        scaling.normalization(self.final_minmax_scaler_cols)
        after_scaling = scaling.data

        drop_cols = DropColumns(after_scaling)
        drop_cols.drop_cols(self.final_cols_to_drop)
        self.result = drop_cols.data


if __name__ == '__main__':
    StreamlitApp()
