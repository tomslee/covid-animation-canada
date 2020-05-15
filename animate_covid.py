#!/usr/bin/python3
"""
Module to generate animations of estimates and actual over time
looking backwards from today.

The data comes from the COVID-19 Canada Open Data Working Group.
Epidemiological Data from the COVID-19 Outbreak in Canada.
https://github.com/ishaberry/Covid19Canada.
"""

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------
import argparse
import os
import os.path
import datetime
# import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ImageMagickFileWriter, FFMpegFileWriter
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import requests
import sqlite3
from scipy.optimize import curve_fit
import bar_chart_race as bcr
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)-8s%(message)s')

# -------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------
DAYS_EXTRAPOLATION = 14
DAYS_ANNOTATE = 6
START_DAYS_OFFSET = 10
FIT_TYPE = "exp"  # "exp" or "poly"
FIT_POINTS = 12
DOWNLOAD_ALWAYS = False
FRAME_COUNT = 42
YSCALE_TYPE = "linear"  # "linear", "log", "symlog", "logit",
INTERPOLATIONS = 3
SMOOTHING_DAYS = 7
# TODO: IMAGEMAGICK_EXE is hardcoded here. Put it in a config file.
IMAGEMAGICK_DIR = "/Program Files/ImageMagick-7.0.9-Q16"
# IMAGEMAGICK_DIR = "/Program Files/ImageMagick-7.0.10-Q16"
# For ImageMagick configuration, see
# https://stackoverflow.com/questions/23417487/saving-a-matplotlib-animation-with-imagemagick-and-without-ffmpeg-or-mencoder/42565258#42565258

# -------------------------------------------------------------------------------
# Set up graphics
# ------------------------------------------------------------------------------
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [7.0, 4.0]
mpl.rcParams['figure.dpi'] = 90
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['animation.convert_path'] = IMAGEMAGICK_DIR + "/magick.exe"
mpl.rcParams['animation.ffmpeg_path'] = IMAGEMAGICK_DIR + "/ffmpeg.exe"
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['legend.fontsize'] = 'large'
# mpl.rcParams['figure.titlesize'] = 'medium'
sns.set()
sns.set_palette("muted")


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def download_data(download_always=DOWNLOAD_ALWAYS):
    """
    Download data from the COVID-19 Canada Open Data Working Group.
    Only download once a day (unless DOWNLOAD_ALWAYS is set) to avoid
    unnecessary refetching of the same data.

    Returns True if a download was performed, False otherwise
    """
    url = ("https://docs.google.com/spreadsheets/d/"
           "1D6okqtBS3S2NRC7GFVHzaZ67DuTw7LX49-fqSLwJyeo/export?format=xlsx")
    if os.path.exists("canada.xlsx"):
        update_time = datetime.date.fromtimestamp(
            os.path.getmtime("canada.xlsx"))
    else:
        update_time = None
    if download_always or datetime.date.today() != update_time:
        logging.info("Downloading spreadsheet...")
        response = requests.get(url)
        with open("canada.xlsx", "wb") as output:
            output.write(response.content)
        return True
    else:
        if os.path.exists("canada.db3"):
            return False
        else:
            return True


def xls_to_db3():
    """
    Read the spreadsheet and store in a database
    """
    if os.path.exists("canada.db3"):
        os.remove("canada.db3")
    dbconn = sqlite3.connect("canada.db3")
    for sheet in ["Cases", "Recovered"]:
        logging.info(f"Reading spreadsheet sheet {sheet}...")
        df = pd.read_excel("canada.xlsx",
                           sheet_name=sheet,
                           skiprows=range(0, 3),
                           index_col=0)
        logging.info(f"Writing spreadsheet sheet {sheet} into sqlite table...")
        df.to_sql(sheet, dbconn, index=True)


class Plot():
    """
    Generic Plot class.
    There's nothing here yet, but it will probably fill up as more plots
    are added
    """
    def output(self, anim, plt, dataset, output):
        """
        Generic output functions
        """
        logging.info("Writing output...")
        filename = "covid_{}.{}".format(dataset.lower(), output)
        if output == "mp4":
            writer = FFMpegFileWriter(fps=10, bitrate=1800)
            anim.save(filename, writer=writer)
        elif output == "gif":
            writer = ImageMagickFileWriter()
            anim.save(filename, writer=writer)
        else:
            plt.show()


class DataSet():
    """
    The data set to plot.
    """
    def __init__(self):
        """
        Generic dataset initializations
        """
        self.dbconn = sqlite3.connect("canada.db3")
        self.population = {
            "NL": 521365,
            "PEI": 158158,
            "Nova Scotia": 977457,
            "New Brunswick": 779993,
            "Quebec": 8537674,
            "Ontario": 14711827,
            "Manitoba": 1377517,
            "Saskatchewan": 1181666,
            "Alberta": 4413146,
            "BC": 5110917,
            "Yukon": 41078,
            "NWT": 44904,
            "Nunavut": 39097,
            "Repatriated": 10000000,
        }

    def interpolate(self, df):
        """
        Interpolate INTERPOLATIONS between every row in df

        Taking from
        https://medium.com/dunder-data/create-a-bar-chart-race-animation-in-python-with-matplotlib-477ed1590096
        This method assumes df has an index with dates in it, called "date".
        This function just repeats the date entry, which is fine when it is
        just used as a label in a bar chart race. However, in a line chart
        animation the result is a series of vertical lines at the repeating 
        date in the index.
        """
        # Make the index an incrementing integer starting at zero
        # The existing index gets turned into a column
        df = df.reset_index()
        # Multiply each index by INTERPOLATIONS
        df.index = df.index * INTERPOLATIONS
        # reindex to add additional rows in between
        max_index = df.index[-1] + 1
        df = df.reindex(range(max_index))
        # Interpolate the date column using date_range
        df["date"] = pd.date_range(start=df["date"].min(),
                                   end=df["date"].max(),
                                   periods=max_index)
        # Now use the date column as index again
        df = df.set_index("date")
        # rank each row
        df_rank = df.rank(axis=1, method="first")
        # interpolate missing values in df and df_rank
        df = df.interpolate()
        df_rank = df_rank.interpolate()
        # print("\nIn interpolate")
        # print(df.tail(INTERPOLATIONS + 1))
        return df, df_rank

    def smooth(self, df, degree=SMOOTHING_DAYS):
        """
        Use the pandas "rolling" method
        """
        for column in df.columns:
            df[column] = df[column].rolling(SMOOTHING_DAYS,
                                            win_type="gaussian",
                                            center=True).mean(std=2)
        # print("\nIn smooth")
        # print(df.tail(INTERPOLATIONS + 1))
        return df

    def smooth_manual(self, df, degree=SMOOTHING_DAYS):
        """
        DEPRECATED: now use built-in dataframe rolling method, above.
        Keeping this just in case it is useful.
        Smooth the values in df
        Taken from
        https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
        df has a date index. All columns are numeric.
        """
        # Compute the weights
        window = degree * 2 - 1
        weight = np.array([1.0] * window)
        weight_gauss = []
        for i in range(window):
            i = i - degree + 1
            frac = i / float(window)
            gauss = 1 / (np.exp((4 * frac)**2))
            weight_gauss.append(gauss)
        weight = np.array(weight_gauss) * weight
        for column in df.columns:
            unsmoothed = df[column].to_list()
            smoothed = [0.0] * (len(df) - window)
            for i, _ in enumerate(smoothed):
                try:
                    df[column][i] = sum(
                        np.array(unsmoothed[i:i + window]) *
                        weight) / sum(weight)
                except ValueError as e:
                    logging.error(e)
                    df[column][i] = float("NaN")
            # df[column] = smoothed
        print("\nIn smooth")
        print(df.tail())
        return df


class Provinces(DataSet):
    """
    Plot an animated bar chart of the provinces totals
    """
    def __init__(self, output=None, plot_type="provinces"):
        """
        Initialize class variables and call what needs to be called.
        """
        super().__init__()
        self.plot_type = plot_type
        self.output = output
        self.provinces = []
        # Populations from
        # https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901
        df = self._prep()
        self.plot_bcr(df)
        # self.plot_manual(df)

    def _prep(self):
        """
        Take the raw dataframe and aggregate it so it can be used by this plot.
        """
        # as_index False yields three columns, like a SQL Group By
        sql_cases = """
        SELECT date_report, province, 1 as cases
        FROM Cases
        """
        df_cases = pd.read_sql(
            sql_cases,
            self.dbconn,
            parse_dates={"date_report": {
                "format": "%Y-%m-%d %H:%M:%S"
            }})
        sql_recovered = """
        SELECT date_recovered, province, cumulative_recovered
        FROM Recovered
        """
        df_recovered = pd.read_sql(
            sql_recovered,
            self.dbconn,
            parse_dates={"date_report": {
                "format": "%Y-%m-%d %H:%M:%S"
            }})
        self.provinces = df_cases["province"].unique()
        # df_cases = df_cases[["date_report", "province"]]
        # Pivot the table on provinces, using the date as index
        # so that there is a row for each date
        df_cases = pd.pivot_table(df_cases,
                                  values="cases",
                                  columns=["province"],
                                  index="date_report",
                                  fill_value=0,
                                  aggfunc=np.size)
        df_recovered = pd.pivot_table(df_recovered,
                                      values="cumulative_recovered",
                                      columns=["province"],
                                      index="date_recovered",
                                      fill_value=0,
                                      aggfunc=np.max)
        print("df_cases")
        print(df_cases["Alberta"].tail())
        print("df_recovered")
        print(df_recovered.tail())
        # For cumulative data rather than daily totals, replace the
        # values in each column with cumulative totals using "expanding"
        for province in self.provinces:
            df_cases[province] = df_cases[province].to_frame().expanding(
                1).sum()
        # join using dates (indexes)
        df_join = df_cases.join(df_recovered,
                                lsuffix='_cases',
                                rsuffix='_recovered')
        for province in self.provinces:
            df_join[province] = df_join[f"{province}_cases"] - df_join[
                f"{province}_recovered"]
        df_join.reset_index(inplace=True)
        df_join.rename(columns={"date_report": "date"}, inplace=True)
        print("df_join")
        print(df_join.tail())
        # df_cases.reset_index(inplace=True)
        # df_cases.rename(columns={"date_report": "date"}, inplace=True)
        # df_cases["date"] = [
        # datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date()
        # for x in df_cases["date"]
        # ]
        # for province in self.provinces:
        # df_cases[province] = 1000000.0 * (df_cases[province] /
        # self.population[province])
        # df_cases["day"] = df_cases.index
        # return df_cases
        columns = ["date"]
        columns.extend(self.provinces)
        df_join = df_join[columns]
        df_join = df_join.fillna(0)
        df_join.set_index("date", inplace=True)
        return df_join

    def plot_bcr(self, df):
        """
        Use the bar_chart_race package to do the plot
        """
        logging.info("Bar Chart Race...")
        fig, ax = plt.subplots()
        bcr.bar_chart_race(
            df=df,
            filename='covid_provinces.mp4',
            orientation='h',
            sort='desc',
            label_bars=True,
            use_index=True,
            steps_per_period=10,
            period_length=500,
            figsize=(6.5, 3.5),
            # cmap="dark24",
            cmap=sns.color_palette(),
            title='Covid-19 Cases Per Million in Canadian Provinces',
            bar_label_size=7,
            tick_label_size=3,
            period_label_size=12,
            fig=fig)

    def plot_manual(self, df):
        """
        Set up the arrays and figure, and call FuncAnimation
        """
        logging.info("Manual Bar Chart Race...")
        fig, ax = plt.subplots()
        plt.title("Work in progress")
        # plt.xlabel("Cases per million")
        color_dict = dict([(p,
                            sns.color_palette()[x % len(sns.color_palette())])
                           for x, p in enumerate(self.provinces)])
        anim = FuncAnimation(fig,
                             self.plot_frame,
                             frames=range(len(df)),
                             fargs=[ax, df, color_dict],
                             interval=500,
                             repeat=False,
                             repeat_delay=1500,
                             save_count=1500)
        Plot().output(anim, plt, self.__class__.__name__, self.output)

    def plot_frame(self, i, ax, df, color_dict):
        """
        Function called from animator to generate frame i of the animation
        """
        ax.clear()
        series = df.iloc[i].sort_values(ascending=True).tail(5)
        colors = [color_dict[x] for x in series.index]
        # logging.info(series)
        ax.barh(series.index, series, color=colors)
        ax.text(1,
                0.2,
                df.index[i].strftime("%b %d"),
                transform=ax.transAxes,
                color='#777777',
                size=24,
                ha='right',
                weight=800)


class GrowthRate(DataSet):
    """
    Plot the percentage change in daily new cases
    """
    def __init__(self, output="", plot_type="growth"):
        """
        Initialize class variables and call what needs to be called.
        """
        super().__init__()
        self.plot_type = plot_type
        # The query uses ROW_NUMBER -1 to start at zero.
        # This matches dataframe behaviour
        sql = """
        SELECT "date", "growth_rate" FROM
        (
        SELECT "date", "new_cases", "cumulative"
        , (100.0 * "new_cases"/(LAG("cumulative", 1, 0) OVER (ORDER BY
        "date")))
        as "growth_rate"
        FROM
            ( SELECT "date"
            , "new_cases"
            , ROW_NUMBER() OVER (ORDER BY "date") -1 as "day"
            , SUM("new_cases") OVER
            (ORDER BY "date" RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            as "cumulative"
            FROM (
                SELECT date_report as "date"
                , count(*) as "new_cases"
                FROM Cases
                GROUP BY date_report
            ) T1
        ) T2
        ) T3
        """
        df = pd.read_sql(sql,
                         self.dbconn,
                         index_col="date",
                         parse_dates={"date": {
                             "format": "%Y-%m-%d %H:%M:%S"
                         }})
        df = df.fillna(0.0)
        print("In init")
        print(df.tail())
        print(df.dtypes)
        self.output = output
        # Interpolate before smoothing!
        df, df_rank = self.interpolate(df)
        df = self.smooth(df)
        self.plot(df)

    def plot(self, df):
        """
        Plot it.

        df has a date index. All columns are numeric.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        fig, ax = plt.subplots()
        anim = FuncAnimation(fig,
                             self.next_frame,
                             frames=len(df),
                             fargs=[ax, df],
                             interval=50,
                             repeat=False,
                             repeat_delay=1500,
                             save_count=1500)
        Plot().output(anim, plt, self.__class__.__name__, self.output)

    def next_frame(self, i, ax, df):
        """
        Function called from animator to generate frame i of the animation
        """
        # logging.info(f"Frame {i}:")
        ax.clear()
        for column in df.columns:
            y = df[column].to_list()
            y[(i + 1):] = [None] * (len(y) - i - 1)
            ax.plot(df.index, y)
        ax.set_xlim(left=min(df.index), right=max(df.index))
        ax.set_ylim(bottom=0, top=df.max().max())
        ax.set_ylabel("Percentage growth in total cases")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


class CumulativeCases(DataSet):
    """
    Plot cumulative cases
    """
    def __init__(self,
                 output="",
                 plot_type="cases",
                 frame_count=FRAME_COUNT,
                 fit_points=FIT_POINTS):
        """
        Initialize the class variables and call what needs to be called.
        The dataframe "data" has a row for each case.
        It must have the following columns:
        - "date_report": the date a case is reported
        """
        super().__init__()
        # The query uses ROW_NUMBER -1 to start at zero.
        # This matches dataframe behaviour
        sql = """
        SELECT "date"
        , "new_cases"
        , ROW_NUMBER() OVER (ORDER BY "date") -1 as "day"
        , SUM("new_cases") OVER
        (ORDER BY "date" RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        as "cumulative"
        FROM (
            SELECT date_report as "date"
            , count(*) as "new_cases"
            FROM Cases
            GROUP BY date_report
        ) T1
        """
        self.data = pd.read_sql(
            sql,
            self.dbconn,
            parse_dates={"date": {
                "format": "%Y-%m-%d %H:%M:%S"
            }})
        self.output = output
        self.frame_count = frame_count
        self.plot_type = plot_type
        self.fit_points = fit_points
        self._prep()
        self.smooth()
        self.plot()

    def _prep(self):
        """
        Take the raw dataframe and aggregate it so it can be used by this plot.
        """
        # self.data = self.data["date_report"].to_frame().groupby(
        # "date_report").apply(lambda x: x.count())
        # self.data has an index named date_report and a single column, named
        # date_report.
        # Fix this up to give a default index and a column "new_cases"
        # self.data.rename(columns={"date_report": "new_cases"}, inplace=True)
        # self.data.reset_index(inplace=True)
        # self.data.rename(columns={"date_report": "date"}, inplace=True)
        # self.data["day"] = self.data.index
        # self.data["cumulative"] =
        # self.data["new_cases"].to_frame().expanding(
        # 1).sum()

    def smooth(self):
        self.data["shift1"] = self.data["cumulative"].shift(1)
        self.data["shift2"] = self.data["cumulative"].shift(2)
        self.data["three_day_moving_average"] = (self.data["cumulative"] +
                                                 self.data["shift1"] +
                                                 self.data["shift2"]) / 3.0
        self.data["cumulative"] = self.data["three_day_moving_average"]

    def plot(self):
        """
        Plot the trend of cumulative cases, observed at
        earlier days, evolving over time.
        """

        most_current_day = self.data["day"].max()
        self.multifit(most_current_day)
        start_day = self.data["day"].min() + START_DAYS_OFFSET
        df1 = self.data[self.data["day"] >= start_day].copy()

        # initial plot
        fig = plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Cumulative cases")
        plt.title("Predicting the present: Covid cases in Canada")
        axis = plt.gca()
        plt.yscale(YSCALE_TYPE)
        axis.set_xlim(left=df1["day"].min(), right=df1["day"].max())
        if YSCALE_TYPE == "log":
            axis.set_ylim(bottom=10, top=df1["cumulative"].max() * 1.5)
        else:
            axis.set_ylim(bottom=0, top=df1["cumulative"].max() * 1.5)
        axis.plot(df1["day"], df1["cumulative"], "o", markersize=8, alpha=0.8)
        axis.plot(df1["day"],
                  df1["cumulative"],
                  "o",
                  markersize=8,
                  color=sns.color_palette()[0],
                  alpha=0.4)
        axis.text(
            0.025,
            0.96,
            "",
            bbox={
                "facecolor": "white",  # 'facecolor': sns.color_palette()[7],
                'alpha': 0.8,
                'pad': 8
            },
            verticalalignment="top",
            transform=axis.transAxes,
            fontsize=11,
            alpha=0.8)
        axis.xaxis.set_major_locator(ticker.IndexLocator(base=7, offset=0))
        xlabels = [
            df1.iloc[i]["date"].strftime("%b %d")
            for i in range(0, len(df1), 7)
        ]
        axis.set_xticklabels(xlabels)
        anim = FuncAnimation(fig,
                             self.next_frame,
                             frames=np.arange(INTERPOLATIONS *
                                              self.frame_count),
                             fargs=[axis, most_current_day, df1],
                             interval=200,
                             repeat=True,
                             repeat_delay=2000)
        Plot().output(anim, plt, self.__class__.__name__, self.output)

    def poly_fit(self, x_var, a_factor, b_factor, c_intercept):
        """
        Fitting function for curve fitting (below)
        """
        return a_factor * x_var**2 + b_factor * x_var + c_intercept

    def exp_fit(self, x_var, a_factor, k_exponent, b_intercept):
        """
        Fitting function for curve fitting (below)
        """
        return a_factor * np.exp(x_var * k_exponent) + b_intercept

    def fit_trends(self, popt, observation_day):
        """
        Fit a line through the observations in the data frame
        """
        fit_type = FIT_TYPE
        x_list = self.data["day"].iloc[observation_day -
                                       self.fit_points:observation_day + 1]
        y_list = self.data["cumulative"].iloc[observation_day -
                                              self.fit_points:observation_day +
                                              1]
        norm_x = x_list.min()
        norm_y = y_list.max()
        x_normalized = x_list - norm_x + 1
        y_normalized = y_list / norm_y
        # print(x, y, x_normalized, y_normalized)
        # feed it into scipy, method is one of ‘lm’, ‘trf’, ‘dogbox’}
        popt_exp, pcov_exp = curve_fit(self.exp_fit,
                                       x_normalized,
                                       y_normalized,
                                       p0=popt,
                                       maxfev=10000)
        # logging.info(" *** EXPONENTIAL *** ")
        # logging.info(pcov_exp)
        popt_poly, pcov_poly = curve_fit(
            self.poly_fit,
            x_normalized,
            y_normalized,
            # p0=popt,
            maxfev=10000)
        # logging.info(" *** POLYNOMIAL *** ")
        # logging.info(pcov_poly)
        if fit_type == "exp":
            popt = popt_exp
            pcov = pcov_exp
        elif fit_type == "poly":
            popt = popt_poly
            pcov = pcov_poly
        return fit_type, popt, pcov

    def add_series_from_fit(self, popt, observation_day, fit_type="exp"):
        """
        Given a fitted function for a given observation day,
        use it to add a column to the dataframe "data" with
        values of that function for the whole data range.
        """
        norm_x = self.data["day"].loc[observation_day - self.fit_points:].min()
        norm_y = self.data["cumulative"].loc[observation_day - self.
                                             fit_points:observation_day].max()
        if fit_type == "exp":
            for day in range(len(self.data)):
                self.data.loc[day, "fit_{}".format(observation_day)] = \
                        int(self.exp_fit((day - norm_x + 1), *popt) * norm_y)
        elif fit_type == "poly":
            for day in range(len(self.data)):
                self.data.loc[day, "fit_{}".format(observation_day)] = \
                        int(self.poly_fit((day - norm_x + 1), *popt) * norm_y)

    def double_time(self, data, observation_date):
        """
        Compute the doubling time of cases at a given observation date
        """
        obs_day = data.loc[observation_date, "day"]
        y_half = 0.5 * data.loc[observation_date, "cumulative"]
        x_0 = data[data["cumulative"] <= y_half]["day"].max()
        x_1 = data[data["cumulative"] >= y_half]["day"].min()
        y_0 = data[data["cumulative"] <= y_half]["cumulative"].max()
        y_1 = data[data["cumulative"] >= y_half]["cumulative"].min()
        x_half = x_0 + (x_1 - x_0) * (y_half - y_0) / (y_1 - y_0)
        double_days = obs_day - x_half
        print(x_0, x_1, x_half, y_0, y_1, y_half, obs_day, double_days)
        return double_days

    def multifit(self, most_current_day):
        """
        Call fit_trends and add_series_from_fit for each day, adding
        a column for each day's trend to the dataframe.
        """
        popt = (3, 0.01, -6)
        for frame in range(self.frame_count):
            observation_day = most_current_day - self.frame_count + frame + 1
            fit_type, popt, pcov = self.fit_trends(
                popt, observation_day=observation_day)
            self.add_series_from_fit(popt,
                                     observation_day=observation_day,
                                     fit_type=fit_type)
            # pred_value = \
            # int(self.data['fit_{}'.format(observation_day)].max())
            # logging.info(f"Prediction {frame}: {pred_value}")

    def next_frame(self, i, axis, most_current_day, data):
        """
        Function called from animator to generate frame i of the animation.
        """
        # (texts, lines, axis) = artists
        texts = [
            child for child in axis.get_children()
            if isinstance(child, mpl.text.Text)
        ]
        lines = axis.get_lines()
        interpolate = (i % INTERPOLATIONS) / INTERPOLATIONS
        observation_day = most_current_day - self.frame_count + int(
            i / INTERPOLATIONS) + 1
        observation_date = data[data["day"] == observation_day].iloc[0]["date"]
        expected_current_value = round(
            int(data["fit_{}".format(observation_day)].max()), -3)
        if interpolate != 0 and observation_day < most_current_day:
            # Between to data points. Interpolate
            yfit_lower = data["fit_{}".format(observation_day)].to_list()
            yfit_upper = data["fit_{}".format(observation_day + 1)].to_list()
            yfit = [
                yfit_lower[j] + interpolate * (yfit_upper[j] - yfit_lower[j])
                for j in range(len(yfit_lower))
            ]
        else:
            yfit = data["fit_{}".format(observation_day)].to_list()
        caption = "\n".join(
            ("Following the trend of the {} days before {},".format(
                self.fit_points, observation_date.strftime("%b %d")),
             "we could have expected {} thousand".format(
                 int(expected_current_value / 1000)),
             "Covid-19 cases in Canada by {}.".format(
                 data["date"].max().strftime("%b %d"))))
        texts[0].set_text(caption)
        yobs = data["cumulative"].to_list()
        # Set yobs to None for day > observation_day and yfit to None for day
        # <= observation_day
        yobs = [
            yobs[i] if i <= (observation_day - data["day"].min()) else None
            for i in range(len(yobs))
        ]
        yfit = [
            yfit[i] if i > (observation_day - data["day"].min()) else None
            for i in range(len(yfit))
        ]
        lines[0].set_ydata(yobs)
        lines[1].set_ydata(yfit)


def parse_args():
    """
    Define, read and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Animate some Covid-19 data.",
                                     usage="%(prog)s [options]",
                                     fromfile_prefix_chars='@')
    parser.add_argument(
        "-o",
        "--output",
        metavar="output",
        action="store",
        type=str,
        default="",
        help="output the animation to the window or as a file; [gif] or mp4",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="dataset",
        action="store",
        type=str,
        default="cases",
        help="data set to plot; [cases] or growth or provinces")
    args = parser.parse_args()
    return args


def read_config(args):
    """
    Take the command line options and read a user-defined config file.
    Together these define how the program runs.
    """
    return "config"


def main():
    """
    Entry point.
    """
    logging.info("Starting...")
    args = parse_args()
    # config = read_config(args)
    if download_data():
        xls_to_db3()
    if args.dataset == "cases":
        CumulativeCases(args.output)
    elif args.dataset == "growth":
        GrowthRate(args.output)
    elif args.dataset == "provinces":
        Provinces(args.output)
    else:
        print("Unknown dataset: {}".format(args.plot))


if __name__ == '__main__':
    main()
