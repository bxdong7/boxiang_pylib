from typing import Tuple, Union, List, Optional
import pandas as pd
from datetime import date, timedelta
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F

def convert_to_tuple(input: Union[str, List[str]]) -> str:
    """
    Convert an input string or list to a tuple
    """
    if type(input) == str:
        input = [input]

    input = [f"'{v}'" for v in input]

    s = "(" + ", ".join(input) + ")"
    return s


def get_ts_data(
        spark: SparkSession,
        game: Union[str, List[str]],
        market: Union[str, List[str]],
        user_source_type_cd: Union[str, List[str]],
        source: Optional[Union[str, List[str]]] = None,
        channel: Optional[Union[str, List[str]]] = None,
        country: Optional[Union[str, List[str]]] = None,
        target_var: str = 'LTV',
        by: str = 'DAY',
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a etl frame that is indexed by date/week/month, and includes LTV, SPEND, CPI, INSTALL, REVENUE (NET), ROAS, FEATURING.
    If target_var is ARPI, also includes ARPI (ARPI_001, ARPI_003, ARPI_007, ARPI_014, ARPI_030, ARPI_060, ARPI_090, ARPI_180, ARPI_270)

    Args:
        spark: a SparkSession object
        game: can be a single game or a list of games
        market: IT or GO or both
        user_source_type_cd: US, MK or [US, MK]
        source (Optional): SKAN or Non-SKAN or both
        channel (Optional)
        country (Optional): full country name. If market includes IT, do not include country
        target_var: ARPI (ARPI_001, ARPI_003, ARPI_007, ARPI_014, ARPI_030, ARPI_060, ARPI_090, ARPI_180, ARPI_270)
                    LTV
                    CPI
                    INSTALL
                    REVENUE
                    ROAS
        by: DAY/WEEK/MONTH
        start_dt (Optional): the start dt of etl loading
        end_dt (Optional): the end dt of etl loading

    Returns:
        a pd.DataFrame without NULL values. For missing periods, it will be all 0s
    """
    # decide table
    if target_var.startswith('ARPI'):
        table = "pr_analytics_agg.fact_promotion_expense_daily"
    else:
        table = "ua.skan_performance_campaign_ltv"

    # construct where statement
    if table == "pr_analytics_agg.fact_promotion_expense_daily":
        mvp_stmt = "MVP_CAMPAIGN_TYPE = 'New Installs'"
    else:
        mvp_stmt = None
    game_stmt = f"APPLICATION_FAMILY_NAME in {convert_to_tuple(game)}"
    market_stmt = f"MARKET_CD in {convert_to_tuple(market)}"
    type_stmt = f"USER_SOURCE_TYPE_CD in {convert_to_tuple(user_source_type_cd)}"
    if ~target_var.startswith('ARPI') and source is not None:
        source_stmt = f"SOURCE in {convert_to_tuple(source)}"
    else:
        source_stmt = None
    if channel is not None:
        channel_stmt = f"CHANNEL_NAME in {convert_to_tuple(channel)}"
    else:
        channel_stmt = None
    if 'IT' not in convert_to_tuple(market) and country is not None:
        country_stmt = f"COUNTRY_NAME in {convert_to_tuple(country)}"
    else:
        country_stmt = None
    if start_dt is not None:
        start_stmt = f"CALENDAR_DT >= '{start_dt}'"
    else:
        start_stmt = None
    if end_dt is not None:
        end_stmt = f"CALENDAR_DT <= '{end_dt}'"
    else:
        end_stmt = None
    where_stmts = [mvp_stmt, game_stmt, market_stmt, type_stmt, source_stmt, channel_stmt, country_stmt, start_stmt,
                   end_stmt]
    where_stmts = [stmt for stmt in where_stmts if stmt is not None]
    where_stmt = " AND ".join(where_stmts)

    # construct select stmt
    if by == 'DAY':
        date_stmt = f"CALENDAR_DT as DAY"
    elif by == 'WEEK':
        date_stmt = "date_add(CALENDAR_DT, -WEEKDAY(CALENDAR_DT)) as WEEK"
    else:
        date_stmt = "date(date_trunc('MONTH', CALENDAR_DT)) as MONTH"
    if table == "pr_analytics_agg.fact_promotion_expense_daily":
        def get_arpi_query(day: int) -> str:
            sql = f"""
            if (CALENDAR_DT > date_sub(current_date(), {day}), 
                null,
                sum(
                    case 
                        when APPLICATION_FAMILY_NAME = 'Harry Potter' then REVS_DAY_{day:0>3d}_AMT*0.616
                        when APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then REVS_DAY_{day:0>3d}_AMT*0.525
                        when APPLICATION_FAMILY_NAME = 'DC Heroes & Villains' then REVS_DAY_{day:0>3d}_AMT*0.546
                        else REVS_DAY_{day:0>3d}_AMT*0.7 
                    end
                )   
            ) as IAP_REVS_DAY_{day:0>3d}_AMT,
            if (CALENDAR_DT > date_sub(current_date(), {day}), 
                null,
                sum(
                    case 
                        when APPLICATION_FAMILY_NAME = 'Harry Potter' then AD_REVS_DAY_{day:0>3d}_AMT*0.88
                        when APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then AD_REVS_DAY_{day:0>3d}_AMT*0.75
                        when APPLICATION_FAMILY_NAME = 'DC Heroes & Villains' then AD_REVS_DAY_{day:0>3d}_AMT*0.78
                        else AD_REVS_DAY_{day:0>3d}_AMT
                    end
                )   
            ) as AD_REVS_DAY_{day:0>3d}_AMT,
            if (CALENDAR_DT > date_sub(current_date(), {day}), 
                null,
                sum(
                    case 
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' and APPLICATION_FAMILY_NAME = 'Harry Potter' then subscriptions_revs_day_{day:0>3d}_amt*0.616 
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then subscriptions_revs_day_{day:0>3d}_amt*0.525
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' then subscriptions_revs_day_{day:0>3d}_amt *0.7
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' and APPLICATION_FAMILY_NAME = 'Harry Potter' then subscriptions_revs_day_{day:0>3d}_amt*0.748
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then subscriptions_revs_day_{day:0>3d}_amt*0.638
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' then subscriptions_revs_day_{day:0>3d}_amt *0.85
                        when MARKET_CD = 'IT' and CALENDAR_DT < date_add(current_date(),-366) and APPLICATION_FAMILY_NAME = 'Harry Potter' then subscriptions_revs_day_{day:0>3d}_amt*0.748 
                        when MARKET_CD = 'IT' and CALENDAR_DT < date_add(current_date(),-366) and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then subscriptions_revs_day_{day:0>3d}_amt*0.638
                        when MARKET_CD = 'IT' and CALENDAR_DT < date_add(current_date(),-366) then subscriptions_revs_day_{day:0>3d}_amt*0.85
                        when MARKET_CD = 'IT' and CALENDAR_DT >= date_add(current_date(),-366) and APPLICATION_FAMILY_NAME = 'Harry Potter' then subscriptions_revs_day_{day:0>3d}_amt*0.616 
                        when MARKET_CD = 'IT' and CALENDAR_DT >= date_add(current_date(),-366) and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then subscriptions_revs_day_{day:0>3d}_amt*0.525
                        when MARKET_CD = 'IT' and CALENDAR_DT >= date_add(current_date(),-366) then subscriptions_revs_day_{day:0>3d}_amt*0.7
                        else 0
                    end
                )
            ) as SUB_REVS_DAY_{day:0>3d}_AMT
            """
            return sql

        arpi_stmt1 = ",\n".join([get_arpi_query(day) for day in [1, 3, 7, 14, 30, 60, 90, 180, 270]])
        arpi_stmt2 = ", ".join([
                                   f"sum(IAP_REVS_DAY_{day:0>3d}_AMT + AD_REVS_DAY_{day:0>3d}_AMT + SUB_REVS_DAY_{day:0>3d}_AMT) / sum(INSTALL) as ARPI_{day:0>3d}"
                                   for day in [1, 3, 7, 14, 30, 60, 90, 180, 270]])
        ltv_stmt = """
            sum(
                case 
                    when APPLICATION_FAMILY_NAME = 'Harry Potter' then LTV_365_LASTEST_VAL*0.616
                    when APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then LTV_365_LASTEST_VAL*0.525
                    when APPLICATION_FAMILY_NAME = 'DC Heroes & Villains' then LTV_365_LASTEST_VAL*0.546
                    else LTV_365_LASTEST_VAL*0.7 
                end
            ) as IAP_LTV,
            sum(
                case 
                    when APPLICATION_FAMILY_NAME = 'Harry Potter' then AD_LTV_365_LASTEST_VAL*0.88
                    when APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then AD_LTV_365_LASTEST_VAL*0.75
                    when APPLICATION_FAMILY_NAME = 'DC Heroes & Villains' then AD_LTV_365_LASTEST_VAL*0.78
                    else AD_LTV_365_LASTEST_VAL 
                end
            ) as AD_LTV,
            sum(
                ifnull(
                    case 
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' and APPLICATION_FAMILY_NAME = 'Harry Potter' then SUB_LTV_365_LASTEST_VAL*0.616 
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then SUB_LTV_365_LASTEST_VAL*0.525
                        when MARKET_CD = 'GO' and CALENDAR_DT < '2022-01-01' then SUB_LTV_365_LASTEST_VAL *0.7
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' and APPLICATION_FAMILY_NAME = 'Harry Potter' then SUB_LTV_365_LASTEST_VAL*0.748
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then SUB_LTV_365_LASTEST_VAL*0.638
                        when MARKET_CD = 'GO' and CALENDAR_DT >= '2022-01-01' then SUB_LTV_365_LASTEST_VAL *0.85
                        when MARKET_CD = 'IT' and APPLICATION_FAMILY_NAME = 'Harry Potter' then SUB_LTV_365_LASTEST_VAL*0.616 
                        when MARKET_CD = 'IT' and APPLICATION_FAMILY_NAME = 'Jurassic World the Game' then SUB_LTV_365_LASTEST_VAL*0.525
                        when MARKET_CD = 'IT' then SUB_LTV_365_LASTEST_VAL*0.7
                        else 0 
                    end, 
                0)
            ) as SUB_LTV
        """
        install_stmt = "sum(USER_QTY) as INSTALL"
        spend_stmt = "sum(EXPENSE_AMT) as SPEND"

        sql_stmt = f"""
            with tmp_tb as (
                select
                    APPLICATION_FAMILY_NAME,
                    MARKET_CD,
                    USER_SOURCE_TYPE_CD,
                    CHANNEL_NAME,
                    COUNTRY_NAME,
                    CALENDAR_DT,
                    {spend_stmt},
                    {install_stmt},
                    {arpi_stmt1},
                    {ltv_stmt}
                from {table}
                where {where_stmt}
                group by 1, 2, 3, 4, 5, 6
                order by 1, 2, 3, 4, 5, 6
            )
            select 
                {date_stmt},
                {arpi_stmt2},
                sum(IAP_LTV + AD_LTV + SUB_LTV) / sum(INSTALL) as LTV,
                sum(SPEND) as SPEND,
                sum(SPEND) / sum(INSTALL) as CPI,
                sum(INSTALL) as INSTALL,
                sum(IAP_LTV + AD_LTV + SUB_LTV) as REVENUE,
                sum(IAP_LTV + AD_LTV + SUB_LTV) / sum(SPEND) as ROAS
            from tmp_tb
            group by 1
            order by 1
        """
    else:
        sql_stmt = f"""
        select 
            {date_stmt},
            sum(IAP_LTV + AD_LTV + SUB_LTV) / sum(INSTALL) as LTV,
            sum(SPEND) as SPEND,
            sum(SPEND) / sum(INSTALL) as CPI,
            sum(INSTALL) as INSTALL,
            sum(IAP_LTV + AD_LTV + SUB_LTV) as REVENUE,
            sum(IAP_LTV + AD_LTV + SUB_LTV) / sum(SPEND) as ROAS
        from {table}
        where {where_stmt}
        group by 1
        order by 1
        """
    ua_df = spark.sql(sql_stmt)

    # get featuring etl
    where_stmts = [game_stmt, market_stmt, start_stmt, end_stmt]
    where_stmts = [stmt for stmt in where_stmts if stmt is not None]
    where_stmt = " AND ".join(where_stmts)
    sql_stmt = f"""
    with prep1 as (
          select 
            case
                when application_name = 'Genies & Gems' then 'Genies and Gems'
                else application_name
            end as APPLICATION_FAMILY_NAME,
            explode(application_market_cd_list) as MARKET_CD,
            (sequence(to_date(start_dt), to_date(end_dt), interval 1 day)) as CALENDAR_DTS,
            event_category_name
          from jc_crm.calendar_events 
          where calendar_name not in  ('UA','Tests','Consumer Insights','Ads','Marketing','Production')
                 and event_category_name not in ('App Store','ASO Testing','Tests','Pause')
    ),
    prep2 as (
        select 
            APPLICATION_FAMILY_NAME,
            MARKET_CD,
            explode(CALENDAR_DTS) as CALENDAR_DT
        from prep1
        where event_category_name = 'Store Featuring'
    ),
    prep3 as (
        select distinct
            {date_stmt},
            1 as FEATURING
        from prep2
        where {where_stmt}
    )
    select *
    from prep3
    """
    feature_df = spark.sql(sql_stmt)

    # merge ua and feature df
    df = ua_df.join(feature_df, on=by, how='outer')
    df = df.na.fill(0)

    # cut df if target is ARPI
    if target_var.startswith('ARPI'):
        day = int(target_var[-3:])
        cutoff_date = date.today() - timedelta(days=day)
        df = df.where(F.col(by) < cutoff_date)

    # set index
    df = df.toPandas()

    if not df.empty:
        df = df.set_index(by)
        df = df.sort_index()

        # fill in missing dates
        start_dt = df.index.min()
        end_dt = df.index.max()

        if by == 'DAY':
            freq = 'D'
        elif by == 'WEEK':
            freq = 'W-MON'
        else:
            freq = 'MS'
        idx = pd.date_range(start_dt, end_dt, freq=freq).date
        df = df.reindex(idx)
        df = df.fillna(0)

    return df