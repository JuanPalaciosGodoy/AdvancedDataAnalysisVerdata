import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px

def spatial_residual_sum_nbcar(
    df_pd: pd.DataFrame,
    resid: np.ndarray,
    geojson_path: str = "data/colombia_departments.geojson",
    dept_code_col_model: str = "DEPT_CODE",
    dept_code_col_geo: str = "dept_code_hecho",
    dept_name_col_geo: str = "dept_name_hecho",
    title: str = "NB-CAR: sum of absolute residuals per department",
):
    """
    Choropleth map of sum of absolute residuals over time, per department,
    for the NB-CAR model.

    Parameters
    ----------
    df_pd : pandas.DataFrame
        Panel used in the CAR model; must contain DEPT_CODE.
        Must be aligned with 'resid' (same row order).
    resid : np.ndarray
        Residuals per observation (raw or Pearson), same length as df_pd.
    geojson_path : str
        Path to colombia_departments.geojson.
    dept_code_col_model : str
        Column in df_pd with department codes (int).
    dept_code_col_geo : str
        Column in GeoJSON with department codes.
    dept_name_col_geo : str
        Column in GeoJSON with department names.
    title : str
        Plot title.
    """

    df_tmp = df_pd.copy()
    df_tmp = df_tmp.assign(resid=resid)

    # 1) Aggregate |resid| per department across all time
    df_group = (
        df_tmp.groupby(dept_code_col_model, as_index=False)["resid"]
        .agg(sum_abs_resid=lambda x: np.abs(x).sum(),
             mean_abs_resid=lambda x: np.abs(x).mean(),
             rmse_resid=lambda x: np.sqrt(np.mean(x**2)))
    )

    # 2) Read GeoJSON and merge
    gdf_dept = gpd.read_file(geojson_path)
    gdf_dept[dept_code_col_geo] = gdf_dept[dept_code_col_geo].astype(int)

    gdf_merge = gdf_dept.merge(
        df_group,
        left_on=dept_code_col_geo,
        right_on=dept_code_col_model,
        how="left",
    )

    gdf_merge = gpd.GeoDataFrame(gdf_merge, geometry="geometry", crs="EPSG:4326")

    # 3) Choropleth
    vals = gdf_merge["sum_abs_resid"].values
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    fig = px.choropleth(
        gdf_merge,
        geojson=gdf_merge.geometry,
        locations=gdf_merge.index,
        color="sum_abs_resid",
        hover_name=dept_name_col_geo,
        range_color=(vmin, vmax),
        scope="south america",
        title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()

    return gdf_merge
