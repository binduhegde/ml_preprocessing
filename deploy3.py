from drop_columns import DropColumns
from feature_scaling import FeatureScaling
from encoding import Encoding
from missing_values import Imputation
import streamlit as st
import pandas as pd
import numpy as np
np.bool = np.bool_
np.object = object


# to style the heading of every step.
# puts a yellow circle at LHS and the heaidng will overlap
# just for aesthetics
def style_heading(text):
    left = """<div style="position: relative; font-family: Arial, sans-serif; font-size: 20px; display: inline-block; margin-top: 50px; margin-bottom: 50px; margin-left: 5px;">
    <span style="position: absolute; top: 50%; left: 0; transform: translate(-50%, -50%); width: 90px; height: 90px; background-color: #F9C58D; border-radius: 50%;"></span>
    <span style="position: relative; left: 5px;">"""
    right = """</span>
</div>"""
    return left + text + right

# it's called when the user clicked 'review my actions' button
# to style the steps taken by the user. returns html string with css included
# will be callled for each step
def create_styled_div(index, heading, items):
    items_html = ''.join(
        [f"<div style='border-radius: 5px; margin-right: 5px; color: white; padding-left: 3px; padding-right: 3px; font-size: 12px; font-family: IBM Plex Sans, sans-serif; line-height: 1.6; background-color: #ff0066; padding-left: 15px; padding-right: 15px; padding-top: 2px; padding-bottom: 2px;'>{item}</div>" for item in items])
    return f"""
    <div style='padding: 20px; border-radius: 10px; max-width: 600px; position: relative; font-family: Arial, sans-serif; margin-bottom: 20px; margin-left: 5%; box-shadow: 0.1rem 0.1rem 1.5rem rgba(0, 0, 0, 0.3);'>
        <div style='position: absolute; width: 600px; height: 10px; background-color: #F9C58D; align-items: center; left: 0px; top: 0px;'></div>
        <div style='position: absolute; top: 30%; left: 5%; width: 50px; height: 50px; background-color: #F9C58D; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; padding: 5px;'>
            {index}
        </div>
        <div style='margin-left: 15%; margin-top: 10px;'>
            <h3 style='margin: 0;'>{heading}</h3>
        </div>
        <div style='margin-top: 30px; display: flex; flex-wrap: wrap; margin-left: 90px;'>
            {items_html}
        </div>
    </div>
    """

class StreamlitApp:
    def __init__(self) -> None:
        # title of the app with gradient background
        title = '<div style="background: linear-gradient(to right,#F9C58D,#F492F0); color: transparent; color: black; font-family: Helvetica; font-weight: bold; font-size: 40px; text-align: center; padding: 30px; letter-spacing:1px; width: 100%; margin-bottom: 20px;"><h2>Machine Learning Preprocessor</h2></div>'
        st.markdown(title, unsafe_allow_html=True)

        # file uploader button which accepts only csv format
        self.df_file = st.file_uploader('Upload the Dataframe', type=['csv'])

        # waits till the user uploads the file then takes the rest of the actions
        if self.df_file is not None:
            # save the dataframe in self.df
            self.df = pd.read_csv(self.df_file)
            # will display all the options the user can take
            self.get_options()

    def get_options(self):
        # step 1
        self.handle_null_values()
        # step 2
        self.encode_categorical_data()
        # step 3
        self.feature_scaling()
        # step 4
        self.drop_unnecessary_cols()
        # step 5
        # only when the user hits this button,
        # all their chosen actions will be displayed
        # after which the user can proceed to download the preprocessed csv file
        if st.button('review my actions'):
            self.review_actions()

    # returns a df wil rows as the columns of the df with columns as
    # missing values, dtype, mean, median and mode
    def missing_values_disciption(self, columns_with_na):
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
        return na_info_df

    # displays and stores the user's choice of handling the null values
    def handle_null_values(self):
        # Heading
        step1 = style_heading('Step 1: Handling Misisng Values')
        st.markdown(step1, unsafe_allow_html=True)

        # displaying the discription of the columns with missing values
        st.text('These are the description of the columns with missing values')
        # Get columns with at least one missing value
        columns_with_na = [
            col for col in self.df.columns if self.df[col].isnull().any()]
        # get the discription dataframe
        na_info_df = self.missing_values_disciption(columns_with_na)
        # display the discription
        st.dataframe(na_info_df)

        # multiselect for choosing to remove rows with missing values in the selected columns
        self.remove_cols = st.multiselect(
            'I want to remove rows of these columns with missing values', columns_with_na)

        # we can fill the missing values in columns whose dtype is object.
        # so we remove these kinds of columns for filling with mean and median options
        # numerical columns with missing values
        num_na_cols = [
            col for col in columns_with_na if self.df[col].dtype != 'object']
        self.fill_mean_cols = st.multiselect(
            'I want to fill the missing values in these columns with the mean', num_na_cols)

        self.fill_median_cols = st.multiselect(
            'I want to fill the missing values in these columns with the median', num_na_cols)

        self.fill_mode_cols = st.multiselect(
            'I want to fill the missing values in these columns with the mode', columns_with_na)

    # 2 ways: onehot encoding and categorical encoding
    def encode_categorical_data(self):
        # heading
        step2 = style_heading('Step 2: Encoding Categorical Data')
        st.markdown(step2, unsafe_allow_html=True)

        # discription of the categorical features
        st.text('These are the categorical feautures')
        # categorical columns
        cat_cols = self.df.select_dtypes(include='object').columns

        # the discriptive df has only 2 columns namely
        # column name and the no of unique values in that column
        cat_info = {
            'Column': [],
            'Unique items': []
        }
        for col in cat_cols:
            cat_info['Column'].append(col)
            cat_info['Unique items'].append(self.df[col].nunique())
        cat_info_df = pd.DataFrame(cat_info)
        # display the description
        st.dataframe(cat_info_df)

        # onehot encoding options
        self.onehot_encoding_cols = st.multiselect(
            'I want to one-hot encode these columns', cat_cols)

        # label encoding options
        self.label_encoding_cols = st.multiselect(
            'I want to label-encode these columns', cat_cols)

    # 2 ways here: minmax scaling and standard scaling
    def feature_scaling(self):
        # heading
        step3 = style_heading('Step 3: Feature Scaling')
        st.markdown(step3, unsafe_allow_html=True)

        # description of the numerical columns
        st.text('These are the numerical columns')
        num_cols = self.df.select_dtypes(exclude='object').columns
        # display the description
        st.dataframe(self.df[num_cols].describe())

        # minmax scaler options
        self.minmax_scaler_cols = st.multiselect(
            'I want to Min-Max scale these columns', num_cols)

        # standard scaler options
        self.std_scaler_cols = st.multiselect(
            'I want to Standard scale these columns', num_cols)

    # is called when the user clicks review my actions button
    def review_actions(self):
        # to keep track of the steps
        ind = 1

        # heading
        step5 = style_heading('Step 5: Review your Actions')
        st.markdown(step5, unsafe_allow_html=True)

        # getting the user's selected actions through this func
        self.set_user_pref()

        # The following code will be applied for all the possible actions the user might have taken
        # it mentions that step only if that action will be done to at least one column

        # *-----------REMOVE COLS----------------
        if len(self.final_remove_cols) != 0:
            html = create_styled_div(
                ind, 'Drop missing values in these columns', self.final_remove_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------FIll MEAN----------------
        if len(self.final_fill_mean_cols) != 0:
            html = create_styled_div(
                ind, 'Fill Null Values with Mean', self.final_fill_mean_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------FIll MEDIAN----------------
        if len(self.final_fill_median_cols) != 0:
            html = create_styled_div(
                ind, 'Fill Null Values with Median', self.final_fill_median_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------FIll MODE----------------
        if len(self.final_fill_mode_cols) != 0:
            html = create_styled_div(
                ind, 'Fill Null Values with Mode', self.final_fill_mode_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------ONEHOT ENCODE----------------
        if len(self.final_onehot_encoding_cols) != 0:
            html = create_styled_div(
                ind, 'One-hot Encode these columns', self.final_onehot_encoding_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------LABEL ENCODE----------------
        if len(self.final_label_encoding_cols) != 0:
            html = create_styled_div(
                ind, 'Label-encode these Columns', self.final_label_encoding_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------MINMAX SCALER----------------
        if len(self.final_minmax_scaler_cols) != 0:
            html = create_styled_div(
                ind, 'Min-max-scale these columns', self.final_minmax_scaler_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------STANDARD SCALER----------------
        if len(self.final_std_scaler_cols) != 0:
            html = create_styled_div(
                ind, 'Standard-scale these columns', self.final_std_scaler_cols)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1

        # *-----------COLS TO DROP----------------
        if len(self.final_cols_to_drop) != 0:
            html = create_styled_div(
                ind, 'Drop these Columns', self.final_cols_to_drop)
            st.markdown(html, unsafe_allow_html=True)
            ind += 1
        self.submit_clicked()

    # sets a multiselect to choose which cols to drop
    def drop_unnecessary_cols(self):
        # heading
        step4 = style_heading('Step 4: Drop Unnecessary Columns')
        st.markdown(step4, unsafe_allow_html=True)

        self.cols_to_drop = st.multiselect(
            'I want to drop these columns', self.df.columns)

    # will be called when the user clicks reviws my actions.
    # this func will be called at the end of the revies_actions func
    def submit_clicked(self):
        # setting the user's preferences
        self.set_user_pref()
        # taking all the selected actions
        self.preprocess()

        # converting it to csv file
        csv = self.result.to_csv(index=False).encode('utf-8')
        st.markdown('---')  # Add a horizontal line

        # displaying download button which if pressed, downloads the new csv file
        st.download_button(
            label="Download Preprocessed Data",
            data=csv,
            file_name='preprocessed_data.csv',
            mime='text/csv',)

    # this funciton will sort any confusion with the options the user selected
    # for eg, if the user chose to fill the missing values with mean AND median of Age.
    # this will solve that problem by removing the age item from the list of filling with median
    # because for each step, all the columns can be selected for only one action in that step
    # this function will basically go from top to down removing all the items presented in the 
    # child's list if it also exists in parent's list
    # only the first option in each step will have exactly what the user chose
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

    # to perform the selected operations on df
    def preprocess(self):
        # *-------MISSING VALUES-----------
        imputation = Imputation(self.df)
        imputation.remove_null(self.final_remove_cols)
        imputation.fill_mean(self.final_fill_mean_cols)
        imputation.fill_median(self.final_fill_median_cols)
        imputation.fill_mode(self.final_fill_mode_cols)
        after_imputation = imputation.data

        # *-------ENCODING-----------
        encode_categorical = Encoding(after_imputation)
        encode_categorical.onehot_encoding(self.final_onehot_encoding_cols)
        encode_categorical.label_encoding(self.final_label_encoding_cols)
        after_encoding = encode_categorical.data

        # *-------SCALING-----------
        scaling = FeatureScaling(after_encoding)
        scaling.standard_scaling(self.final_std_scaler_cols)
        scaling.normalization(self.final_minmax_scaler_cols)
        after_scaling = scaling.data

        # *-------DROPING-----------
        drop_cols = DropColumns(after_scaling)
        drop_cols.drop_cols(self.final_cols_to_drop)
        self.result = drop_cols.data
        # finally, the preprocessed df is saved in self.result

if __name__ == '__main__':
    StreamlitApp()
