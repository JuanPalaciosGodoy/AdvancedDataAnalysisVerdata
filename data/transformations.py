import polars as pl
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from data.mappings import DEPT_CODE


def preprocess_data(
    df: pl.DataFrame,
) -> pl.DataFrame:

    # standardize data
    columns = [
        'NUMBER_OF_HOMICIDIO',
        'NUMBER_OF_RECLUTAMIENTO',
        'NUMBER_OF_DESAPARICION',
        'NUMBER_OF_SECUESTRO',
        'S01','S02','S03','S04',
        'S05','S06','S07','S08'
    ]

def plot_acf_pacf(
    df: pl.DataFrame,
    col:str,
    dept_code:int,
) -> pl.DataFrame:

    # filter data by DEPT_CODE
    df_dept = df.filter(pl.col('DEPT_CODE')==dept_code).sort(by='YYYYMM')

    # Extract the time series data as a NumPy array (or Pandas Series if preferred)
    time_series = df_dept[col].to_numpy()

    # Plot ACF
    fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
    plot_acf(time_series, ax=ax_acf, lags=20) # Adjust lags as needed
    ax_acf.set_title(f"Autocorrelation Function (ACF): {DEPT_CODE[str(dept_code)]}")
    plt.show()

    # Plot PACF
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
    plot_pacf(time_series, ax=ax_pacf, lags=20) # Adjust lags as needed
    ax_pacf.set_title(f"Partial Autocorrelation Function (PACF): {DEPT_CODE[str(dept_code)]}")
    plt.show()

def standardize(
    df: pl.DataFrame,
    columns:list=[]
) -> pl.DataFrame:

    """standardize predefined columns. if no columns are defined, this function will attempt to standardize all columns"""

    # define columns that need standardization
    cols = columns if len(columns)>0 else df.columns

    means = df.select([pl.col(c).mean().alias(f"{c}_mean") for c in cols]).row(0)
    stds = df.select([pl.col(c).std().alias(f"{c}_std") for c in cols]).row(0)

    # Standardize the numerical columns
    df_standardized = df.with_columns([
        ((pl.col(c) - means[i]) / stds[i]).alias(f"{c}_STANDARDIZED")
        for i, c in enumerate(cols)
    ])

    return df_standardized