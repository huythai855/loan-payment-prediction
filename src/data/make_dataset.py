from sklearn.model_selection import train_test_split
import pandas as pd
import logging


def write_dataset(df, file_name, out_path):
    df_train, df_test = train_test_split(df, test_size=.2, random_state=None)
    df_train.to_csv(out_path + file_name + "_train.csv", index=False)
    df_test.to_csv(out_path + file_name + "_test.csv", index=False)


def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df


def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df


def preprocess_inputs(df):
    df = df.copy()

    # Drop Loan_ID column
    df = df.drop('Loan_ID', axis=1)

    # Create date/time columns
    for column in ['effective_date', 'due_date', 'paid_off_time']:
        df[column] = pd.to_datetime(df[column])

    df['effective_day'] = df['effective_date'].apply(lambda x: x.day)

    df['due_month'] = df['due_date'].apply(lambda x: x.month)
    df['due_day'] = df['due_date'].apply(lambda x: x.day)

    df['paid_off_month'] = df['paid_off_time'].apply(lambda x: x.month)
    df['paid_off_day'] = df['paid_off_time'].apply(lambda x: x.day)
    df['paid_off_hour'] = df['paid_off_time'].apply(lambda x: x.hour)

    df = df.drop(['effective_date', 'due_date', 'paid_off_time'], axis=1)

    # Fill missing values with column means
    for column in ['past_due_days', 'paid_off_month', 'paid_off_day', 'paid_off_hour']:
        df[column] = df[column].fillna(df[column].mean())

    # Binary encode the Gender column
    df = binary_encode(df, 'Gender', positive_value='male')

    # Ordinal encode the education column
    education_ordering = [
        'High School or Below',
        'college',
        'Bechalor',
        'Master or Above'
    ]
    df = ordinal_encode(df, 'education', ordering=education_ordering)

    # Encode the label (loan_status) column
    label_mapping = {'COLLECTION': 0, 'PAIDOFF': 1, 'COLLECTION_PAIDOFF': 2}
    df['loan_status'] = df['loan_status'].replace(label_mapping)
    return df


def make():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_path = "../../data/raw/loan-payments-data.csv"
    output_path = "../../data/processed/"

    # read the raw dataset
    logger = logging.getLogger(__name__)
    logger.info('Reading Raw Dataset...')
    df = pd.read_csv(input_path)
    df = preprocess_inputs(df)

    # write the processed dataset
    logger.info('Writing Train/Test Datasets...')
    write_dataset(df, 'lpd', output_path)
