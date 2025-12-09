import polars as pl
from data.mappings import DEPT_CODE, SECTOR_ID, ViolentEvent

def read_data(violent_event:str, version:str="v2", replica_ids:list=[1], filters:dict={'is_conflict':True}) -> pl.DataFrame:
    """
    Read data from verdata's parquet files

    Parameters:
    -----------
        violent_event (str):
            corresponds to a violent event category as defined in Verdata. These include: `desaparicion`, `homicidio`, `reclutamiento`, and `secuestro`

        version (str):
            corresponds to Verdata's version of the data

    Returns:
    --------
        (polars.DataFrame):
            dataframe with verdata data for the specified violent event.

    """

    event = ViolentEvent(violent_event).value

    df = pl.DataFrame()
    for i in replica_ids:

        print(f"reading file: verdata-{event}-R{i}.parquet")

        # define file path
        file_path = f"extdata/{event}-{version}.parquet/verdata-{event}-R{i}.parquet"

        # Read the Parquet file into a Polars DataFrame
        df_i = pl.read_parquet(file_path)

        df = pl.concat([df, df_i])

    # apply filters
    for key, value in filters.items():
        df = df.filter(pl.col(key) == value)

    return df

def aggregated_event_data(
        event:str,
        replica_ids:list=[1],
    ) -> pl.DataFrame:
    """
    Aggregate data for given dataframe
    """

    group_columns = ["dept_code_hecho", "yymm_hecho"]
    count_column = "dept_code_hecho"

    # read event data
    df = read_data(violent_event=event, replica_ids=replica_ids)
    df = df.with_columns(pl.col("yymm_hecho").cast(pl.Int64))

    # count number of events
    df_count = df.group_by(group_columns).agg(
            pl.col(count_column).count().alias(f"NUMBER_OF_{event.upper()}")
        )

    # rename variables to coincide with the other dataframes
    df_count = df_count.rename({"dept_code_hecho": "DEPT_CODE", "yymm_hecho": "YYYYMM"}).sort(["DEPT_CODE", "YYYYMM"])

    df_count = df_count.with_columns([
        pl.col("DEPT_CODE").cast(pl.Int32),
        pl.col("YYYYMM").cast(pl.Int64),
    ])

    return df_count

def get_lat_lon_data() -> pl.DataFrame:

    file_path = 'data/coordinates.csv'

    # read coordinates
    df = pl.read_csv(file_path)

    # Convert columns
    df = df.unique(subset=["DEPT_CODE"])[["DEPT_CODE", "DEPT_LON", "DEPT_LAT"]]

    return df.with_columns(
        pl.col('DEPT_CODE').cast(pl.Int32)
    )

def get_population_data() -> pl.DataFrame:

    file_path = 'data/population.csv'

    # read coordinates
    df = pl.read_csv(file_path)[["DEPT_CODE", "YYYY", "POPULATION"]]

    # Convert columns
    df = df.with_columns(
        pl.col('DEPT_CODE').cast(pl.Int32)
    )

    df_monthly = (
        df
        .sort(["DEPT_CODE", "YYYY"])
        # compute next year's POPULATION for each department
        .with_columns(
            pl.col("POPULATION")
            .shift(-1)
            .over("DEPT_CODE")
            .alias("POP_NEXT")
        )
        # assign months 1..12 to every row (one year -> 12 months)
        .with_columns(
            pl.lit(list(range(1, 13))).alias("MONTH")
        )
        .explode("MONTH")
        # linearly interpolate between POPULATION (this year) and POP_NEXT (next year)
        .with_columns(
            (
                pl.col("POPULATION")
                + (pl.col("POP_NEXT").fill_null(pl.col("POPULATION")) - pl.col("POPULATION"))
                * ( (pl.col("MONTH") - 1) / 12.0 )
            ).alias("POPULATION_MONTHLY"),
            (pl.col("YYYY") * 100 + pl.col("MONTH"))
            .cast(pl.Int64)
            .alias("YYYYMM"),
        )
        .select(
            "DEPT_CODE",
            "YYYYMM",
            pl.col("POPULATION_MONTHLY").alias("POPULATION"),
        )
    )

    return df_monthly

def get_panel_data(
    replica_ids:list=[1],
) -> pl.DataFrame:

    # read gdp data
    df_gdp = get_gdp()

    # pivot data to get each sector as column
    df_gdp = df_gdp.pivot(
        on=['SECTOR_ID'],
        index=['DEPT_CODE', 'DEPT_NAME', 'YYYYMM'],
        values='GDP(Bill)(COP)'
    )

    # read event data
    df_panel = pl.DataFrame()
    for event in ['homicidio', 'reclutamiento', 'desaparicion', 'secuestro']:
        df_event = aggregated_event_data(
            event=event,
            replica_ids=replica_ids
        )
        if df_panel.is_empty():
            df_panel = df_event
            continue

        df_panel = df_panel.join(
            df_event,
            on=["DEPT_CODE", "YYYYMM"],
            how="left"
        )

    # join gdp data
    df_panel = df_panel.join(
        df_gdp,
        on=["DEPT_CODE", "YYYYMM"],
        how="left"
    )

    # join coordinates
    df_coord = get_lat_lon_data()
    df_panel = df_panel.join(
        df_coord,
        on=["DEPT_CODE"],
        how="left"
    )

    # join population
    df_population = get_population_data()
    df_panel = df_panel.join(
        df_population,
        on=["DEPT_CODE", "YYYYMM"],
        how="left"
    )

    # convert to date yyyymm
    df_panel = df_panel.with_columns(pl.col('YYYYMM').cast(pl.String).str.strptime(pl.Date, format="%Y%m", strict=False))

    # add month and year as columns (features)
    df_panel = (
        df_panel
        .with_columns([
            pl.col("DEPT_CODE").cast(pl.Int32),
            pl.col("YYYYMM").cast(pl.Date),
        ])
        .with_columns([
            pl.col("YYYYMM").dt.year().alias("year"),
            pl.col("YYYYMM").dt.month().alias("month"),
        ])
        .sort(["DEPT_CODE", "YYYYMM"])
    )

    return df_panel.fill_null(0)

def get_gdp_data(is_secondary_source:bool=True) -> pl.DataFrame:

    if is_secondary_source:
        file_path = f"data/gdp_2.csv"
        years_excluded = ['2016p']
    else:
        file_path = f"data/gdp.csv"
        years_excluded = ['2023p','2024pr']

    # Read the csv file into a Polars DataFrame
    df = pl.read_csv(file_path)

    # adjust data types
    df = df.filter(pl.col('YYYY').is_in(years_excluded).not_())
    return df.with_columns(pl.col("YYYY").cast(pl.Int32))

def get_estimate_gdp_pct(
        is_secondary_source:bool=True,
        new_department_ids:list = [91, 81, 85, 94, 95, 86, 88, 97, 99],
        reference_years:list = [2000, 2001, 2002, 2003],
        estimation_years:list = list(range(1985, 2000))
) -> pl.DataFrame:

    # Read the csv file into a Polars DataFrame
    df = get_gdp_data(is_secondary_source=is_secondary_source)

    # filter by department ids
    df_new_dept = df.filter(pl.col("DEPT_CODE").is_in(new_department_ids))

    # get reference years
    reference_years = df_new_dept.filter(pl.col('YYYY').is_in(reference_years))

    # calculate mean of reference years
    df_ref_mean = reference_years['DEPT_CODE', 'SECTOR_ID', 'GDP(Bill)(COP)'].group_by(['DEPT_CODE', 'SECTOR_ID']).mean()

    # calculate percentage of gdp of each DEPT_CODE per SECTOR_ID
    df_new_dept_pct = (
        df_ref_mean
        .with_columns([
            pl.col("GDP(Bill)(COP)").sum().over(["SECTOR_ID"]).alias("SECTOR_TOTAL_YYYY"),
        ])
        .with_columns([
            (pl.col("GDP(Bill)(COP)") / pl.col("SECTOR_TOTAL_YYYY")).alias("PCT_GDP_SECTOR"),
        ])
    )

    # create dataframe template with all estimation years
    new_dept = [item for item in new_department_ids for _ in range(len(estimation_years))] # repeat new departments for each missing year
    df_yyyy = pl.DataFrame({"YYYY": estimation_years * len(new_department_ids), "DEPT_CODE": new_dept})

    # create dataframe with all estimation years and estimated percentage
    df_pct = df_yyyy.join(df_new_dept_pct, on="DEPT_CODE", how="left")['YYYY','DEPT_CODE','SECTOR_ID','PCT_GDP_SECTOR']

    return df_pct.with_columns(
        pl.col('YYYY').cast(pl.Int64),
        pl.col('DEPT_CODE').cast(pl.Int64),
        pl.col('SECTOR_ID').cast(pl.String),
        pl.col('PCT_GDP_SECTOR').cast(pl.Float64)
    )

def get_gdp_pct(
    is_secondary_source:bool=True,
    department_ids:list = [91, 81, 85, 94, 95, 86, 88, 97, 99],
    years:list = list(range(2000, 2016))
) -> pl.DataFrame:

    # Read the csv file into a Polars DataFrame
    df = get_gdp_data(is_secondary_source=is_secondary_source)

    # filter by department ids
    df_new_dept = df.filter(pl.col("DEPT_CODE").is_in(department_ids))

    # get reference years
    df_years = df_new_dept.filter(pl.col('YYYY').is_in(years))

    # calculate percentage of gdp of each DEPT_CODE per SECTOR_ID
    df_dept_pct = (
        df_years
        .with_columns([
            pl.col("GDP(Bill)(COP)").sum().over(["SECTOR_ID", "YYYY"]).alias("SECTOR_TOTAL_YYYY"),
        ])
        .with_columns([
            (pl.col("GDP(Bill)(COP)") / pl.col("SECTOR_TOTAL_YYYY")).alias("PCT_GDP_SECTOR"),
        ])
    )

    df_pct = df_dept_pct['YYYY','DEPT_CODE','SECTOR_ID','SECTOR_TOTAL_YYYY','PCT_GDP_SECTOR']

    return df_pct.with_columns(
        pl.col('YYYY').cast(pl.Int64),
        pl.col('DEPT_CODE').cast(pl.Int64),
        pl.col('SECTOR_ID').cast(pl.String),
        pl.col('SECTOR_TOTAL_YYYY').cast(pl.Float64),
        pl.col('PCT_GDP_SECTOR').cast(pl.Float64)
    )


def get_gdp() -> pl.DataFrame:
    # first chunk of missing data for New Departments
    df_1 = get_estimate_gdp_pct()
    df_1_total = get_gdp_pct(is_secondary_source=False, department_ids=[99999], years=list(range(1985, 2000)))['YYYY','SECTOR_ID','SECTOR_TOTAL_YYYY']
    df_1 = df_1.join(df_1_total, on=['YYYY','SECTOR_ID'])

    # last chunk of missing data for New Departments
    df_2 = get_estimate_gdp_pct(reference_years=[2012, 2013, 2014, 2015], estimation_years=list(range(2016, 2022)))
    df_2_total = get_gdp_pct(is_secondary_source=False, department_ids=[99999], years=list(range(2016, 2022)))['YYYY','SECTOR_ID','SECTOR_TOTAL_YYYY']
    df_2 = df_2.join(df_2_total, on=['YYYY','SECTOR_ID'])

    # known data for New Departments
    df_3 = get_gdp_pct()

    # concatenate all data for all periods
    df_new_dept = pl.concat([df_1, df_2, df_3], how='align')

    # calculate gdp for New Departments
    df_new_dept = df_new_dept.with_columns((pl.col('SECTOR_TOTAL_YYYY') * pl.col('PCT_GDP_SECTOR')).alias('GDP(Bill)(COP)'))
    df_new_dept = df_new_dept.with_columns((pl.col('GDP(Bill)(COP)') * 1000000000).alias('GDP(COP)'))

    # get GDP for other departments
    df_gdp = get_gdp_data(is_secondary_source=False)

    # Cast GDP as float
    df_gdp = df_gdp.with_columns(
            pl.col('YYYY').cast(pl.Int64),
            pl.col('DEPT_CODE').cast(pl.String),
            pl.col('DEPT_NAME').cast(pl.String),
            pl.col('SECTOR').cast(pl.String),
            pl.col('SECTOR_ID').cast(pl.String),
            pl.col('GDP(Bill)(COP)').cast(pl.Float64),
            pl.col('GDP(COP)').cast(pl.Float64),
        )['YYYY','DEPT_CODE','DEPT_NAME','SECTOR','SECTOR_ID','GDP(Bill)(COP)','GDP(COP)']

    # remove NUEVOS DEPARTAMENTOS since these are added using df_new_dept
    df_gdp = df_gdp.filter(pl.col("DEPT_CODE").is_in(['99999']).not_())

    # add department name and sector name
    df_new_dept = df_new_dept.with_columns(pl.col("DEPT_CODE").cast(pl.String))
    df_new_dept = df_new_dept.with_columns(pl.col("DEPT_CODE").replace(DEPT_CODE).alias("DEPT_NAME"))
    df_new_dept = df_new_dept.with_columns(pl.col("SECTOR_ID").replace(SECTOR_ID).alias("SECTOR"))
    df_new_dept = df_new_dept['YYYY','DEPT_CODE','DEPT_NAME','SECTOR','SECTOR_ID','GDP(Bill)(COP)','GDP(COP)']

    # concatenate results for new departments and the other departments
    df_complete = pl.concat([df_gdp, df_new_dept], how='vertical').sort(by=['YYYY','DEPT_CODE','SECTOR_ID'])

    # change datatype of dept_code
    df_complete = df_complete.with_columns(pl.col("DEPT_CODE").cast(pl.Int32))

    # create monthly data
    df_complete = (
        df_complete
        # sort so "next year" is well-defined
        .sort(["DEPT_CODE", "SECTOR_ID", "YYYY"])
        # get next year's GDP values per DEPT_CODE x SECTOR_ID
        .with_columns(
            pl.col("GDP(Bill)(COP)")
            .shift(-1)
            .over(["DEPT_CODE", "SECTOR_ID"])
            .alias("GDP_Bill_next"),
            pl.col("GDP(COP)")
            .shift(-1)
            .over(["DEPT_CODE", "SECTOR_ID"])
            .alias("GDP_next"),
        )
        # assign months 1..12 to every row (one year -> 12 months)
        .with_columns(
            pl.lit(list(range(1, 13))).alias("MONTH")
        )
        .explode("MONTH")
        # build YYYYMM and linearly interpolate the GDP columns
        .with_columns(
            (pl.col("YYYY") * 100 + pl.col("MONTH"))
            .cast(pl.Int64)
            .alias("YYYYMM"),

            # interpolate annual to monthly: month 1 = current year,
            # month 12 ~ just before next year
            (
                pl.col("GDP(Bill)(COP)")
                + (
                    pl.col("GDP_Bill_next").fill_null(pl.col("GDP(Bill)(COP)"))
                    - pl.col("GDP(Bill)(COP)")
                )
                * ((pl.col("MONTH") - 1) / 12.0)
            ).alias("GDP(Bill)(COP)"),

            (
                pl.col("GDP(COP)")
                + (
                    pl.col("GDP_next").fill_null(pl.col("GDP(COP)"))
                    - pl.col("GDP(COP)")
                )
                * ((pl.col("MONTH") - 1) / 12.0)
            ).alias("GDP(COP)"),
        )
        .select(
            "DEPT_CODE",
            "YYYYMM",
            "DEPT_NAME",
            "SECTOR",
            "SECTOR_ID",
            "GDP(Bill)(COP)",
            "GDP(COP)",
        )
    )

    return df_complete


if __name__ == "__main__":

    # Example 1: desaparicion
    print("=== Desaparicion ===")
    df_des = read_data(violent_event="desaparicion")
    print(f"Data: {len(df_des)}")

    # Example 2: homicidio
    print("=== Homicidio ===")
    df_hom = read_data(violent_event="homicidio")
    print(f"Data: {len(df_hom)}")

    # Example 3: reclutamiento
    print("=== Reclutamiento ===")
    df_rec = read_data(violent_event="reclutamiento")
    print(f"Data: {len(df_rec)}")

    # Example 4: homicidio
    print("=== Secuestro ===")
    df_sec= read_data(violent_event="secuestro")
    print(f"Data: {len(df_sec)}")

    # Example 5: GDP
    print("=== GDP ===")
    df_gdp= get_gdp_data()
    print(f"Data: {len(df_gdp)}")