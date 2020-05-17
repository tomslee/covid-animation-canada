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
logger = logging.getLogger('animate_covid')

# -------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------
DAYS_EXTRAPOLATION = 14
DAYS_ANNOTATE = 6
START_DAYS_OFFSET = 20
FIT_TYPE = "exp"  # "exp" or "poly"
FIT_POINTS = 10
POPT_GUESS = (3, 0.01, -6)
DOWNLOAD_ALWAYS = False
YSCALE_TYPE = "linear"  # "linear", "log", "symlog", "logit",
INTERPOLATION_POINTS = 5
SMOOTHING_DAYS = 7
SMOOTHING_STD = 3
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
        logger.info("Downloading spreadsheet...")
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
        logger.info(f"Reading spreadsheet sheet {sheet}...")
        df = pd.read_excel("canada.xlsx",
                           sheet_name=sheet,
                           skiprows=range(0, 3),
                           index_col=0)
        logger.info(f"Writing spreadsheet sheet {sheet} into sqlite table...")
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
        logger.info("Writing output...")
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
        Interpolate INTERPOLATION_POINTS between every row in df

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
        # Multiply each index by INTERPOLATION_POINTS
        df.index = df.index * INTERPOLATION_POINTS
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
        return df, df_rank

    def smooth(self,
               df,
               column_list="all",
               degree=SMOOTHING_DAYS,
               center=True):
        """
        Use the pandas "rolling" method
        """
        if column_list == "all":
            column_list = df.columns.to_list()
        for column in column_list:
            df[column] = df[column].rolling(
                SMOOTHING_DAYS,
                win_type="gaussian",
                center=center,
                min_periods=0).mean(std=SMOOTHING_STD)
        df.fillna(method='ffill', inplace=True)
        # values at the "top" of df may still be NaN
        df.fillna(0, inplace=True)
        return df

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
        logger.info("Bar Chart Race...")
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
        logger.info("Manual Bar Chart Race...")
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
        # logger.info(series)
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
        ax.clear()
        for column in df.columns:
            y = df[column].to_list()
            y[(i + 1):] = [None] * (len(y) - i - 1)
            ax.plot(df.index, y, lw=4)
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
    def __init__(self, output="", plot_type="cases", fit_points=FIT_POINTS):
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
        , ROW_NUMBER() OVER (ORDER BY "date") -1 as "day"
        , SUM("new_cases") OVER
        (ORDER BY "date" RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        as "cumulative"
        , 0 as "trend_0"
        , 0 as "trend_1"
        FROM (
            SELECT date_report as "date"
            , count(*) as "new_cases"
            FROM Cases
            GROUP BY date_report
        ) T1
        """
        df = pd.read_sql(sql,
                         self.dbconn,
                         index_col="date",
                         parse_dates={"date": {
                             "format": "%Y-%m-%d %H:%M:%S"
                         }})
        # df index is the date.
        # df columns are "day", "cumulative", and "trend"
        # "trend_0" and "trend_1" are NULL, and will be filled in later
        # We keep two trend columns for interpolation
        self.output = output
        self.plot_type = plot_type
        self.fit_points = fit_points
        df = df[df["day"] >= START_DAYS_OFFSET]
        df["day"] = list(range(len(df)))
        df = self.smooth(df,
                         column_list=["cumulative"],
                         degree=3,
                         center=False)
        logger.debug((f"Initial values: index={df.index.values[0]}, "
                      f"day={df['day'].iloc[0]}, "
                      f"cumulative={df['cumulative'].iloc[0]}"))
        trend_lines = self.fit_trend_lines(df)
        self.plot(df, trend_lines)

    def fit_trend_lines(self, df):
        """
        Call fit_trends for each day, adding
        a column for each day's trend to the dataframe.
        """
        logger.info("Fitting trend lines...")
        popt = POPT_GUESS
        trend_lines = []
        for observation_day in range(self.fit_points, len(df)):
            logger.debug((f"Obs day = {observation_day}, "
                          f"date={df.index.values[observation_day]}, "
                          f"popt={popt}"))
            fit_type, popt, pcov = self.fit_trends(
                df,
                x_col="day",
                y_col="cumulative",
                popt=popt,
                observation_day=observation_day)
            trend_lines.append((fit_type, observation_day, popt, pcov))
        logger.debug(f"Length of trend_lines = {len(trend_lines)}")
        return trend_lines

    def fit_trends(self, df, x_col, y_col, popt=None, observation_day=None):
        """
        Fit a line through the observations in the data frame
        """
        fit_type = FIT_TYPE
        fit_min = observation_day - self.fit_points
        fit_max = observation_day
        x = df[x_col].iloc[fit_min:fit_max]
        y = df[y_col].iloc[fit_min:fit_max]
        norm_x = x.min()
        norm_y = y.max()
        x_normalized = x - norm_x + 1
        y_normalized = y / norm_y
        # feed it into scipy, method is one of ‘lm’, ‘trf’, ‘dogbox’}
        popt_exp, pcov_exp = curve_fit(self.exp_fit,
                                       x_normalized,
                                       y_normalized,
                                       p0=popt,
                                       maxfev=10000)
        popt_poly, pcov_poly = curve_fit(self.poly_fit,
                                         x_normalized,
                                         y_normalized,
                                         p0=popt,
                                         maxfev=10000)
        # logger.info(" *** POLYNOMIAL *** ")
        # logger.info(pcov_poly)
        if fit_type == "exp":
            popt = popt_exp
            pcov = pcov_exp
        elif fit_type == "poly":
            popt = popt_poly
            pcov = pcov_poly
        return fit_type, popt, pcov

    def plot(self, df, trend_lines):
        """
        Plot the trend of cumulative cases, observed at
        earlier days, evolving over time.
        """
        # initial plot
        logger.info("Plotting...")
        fig, ax = plt.subplots()
        anim = FuncAnimation(fig,
                             self.next_frame,
                             frames=(INTERPOLATION_POINTS + 1) *
                             (len(df) - self.fit_points),
                             fargs=[ax, df, trend_lines],
                             interval=50,
                             repeat=False,
                             repeat_delay=3000)
        Plot().output(anim, plt, self.__class__.__name__, self.output)

    def next_frame(self, i, ax, df, trend_lines):
        """
        Function called from animator to generate frame i of the animation.
        """
        # Get the objects we are going to update
        ax.clear()

        # Compute the trend_0 and trend_1 columns: the trend on
        # observation_day and on observation_day + 1 where possible
        observation_day = int(i / (INTERPOLATION_POINTS + 1)) + self.fit_points
        df = self.update_trend_columns(df,
                                       i,
                                       x_col="day",
                                       y_col="cumulative",
                                       trend_cols=["trend_0", "trend_1"],
                                       trend_lines=trend_lines,
                                       observation_day=observation_day)
        interpolate = (i %
                       (INTERPOLATION_POINTS + 1)) / (INTERPOLATION_POINTS + 1)
        logger.debug(f"Interpolate = {interpolate}")
        y_obs = df["cumulative"].to_list()
        if interpolate != 0 and observation_day < df["day"].max():
            # Between to data points. Interpolate
            y_fit_0 = df["trend_0"].to_list()
            y_fit_1 = df["trend_1"].to_list()
            y_fit = [
                y_fit_0[j] + interpolate * (y_fit_1[j] - y_fit_0[j])
                for j, _ in enumerate(y_fit_0)
            ]
        else:
            y_fit = df["trend_0"].to_list()
        # Set y_obs to None for day > observation_day and y_fit to None for day
        # <= observation_day
        y_obs = [
            y_obs[i] if i <= (observation_day) else None
            for i, _ in enumerate(y_obs)
        ]
        y_fit = [
            y_fit[i] if i > (observation_day) else None
            for i, _ in enumerate(y_fit)
        ]
        ax.plot(df.index,
                y_obs,
                "-",
                markersize=6,
                lw=3,
                color=sns.color_palette()[0],
                alpha=0.9)
        ax.plot(df.index,
                y_fit,
                "-",
                markersize=6,
                lw=3,
                color=sns.color_palette()[2],
                alpha=0.4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative cases")
        ax.set_title("Predicting the present: Covid cases in Canada")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_yscale(YSCALE_TYPE)
        if YSCALE_TYPE == "log":
            ax.set_ylim(bottom=10, top=df["cumulative"].max() * 1.2)
        else:
            ax.set_ylim(bottom=0, top=df["cumulative"].max() * 1.2)

        # caption
        observation_date = df.index.values[observation_day]
        datestring = pd.to_datetime(str(observation_date)).strftime("%b %d")
        expected_current_value = round(int(df["trend_0"].max()), -3)
        expected = int(expected_current_value / 1000)
        today = df.index.max().strftime("%b %d")
        caption = "\n".join(((f"Following the trend of the {self.fit_points} "
                              f"days before {datestring}"),
                             f"we could have expected {expected} thousand",
                             f"Covid-19 cases in Canada by {today}."))
        ax.text(0.025,
                0.96,
                caption,
                bbox={
                    "facecolor": sns.color_palette()[4],
                    'alpha': 0.2,
                    'pad': 8
                },
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=11,
                alpha=0.8)

    def update_trend_columns(self, df, frame, x_col, y_col, trend_cols,
                             trend_lines, observation_day):
        """
        Given a trend function for a given observation day,
        use it to update two columns of the dataframe "df" with
        values of that function for the whole data range.

        trend_cols is a two-element list of column names
        to hold the two trend columns
        """
        # j chosen so that we do not do a second trend line if we are at
        # the final day in the data frame
        j = 1 - int(observation_day / (len(df) - 1))
        for i in list(range(j + 1)):
            line_index = observation_day + i - self.fit_points
            logger.debug((f"Update {frame}.{i}, day {observation_day}: "
                          f"trend_line[{line_index}]"))
            trend_line = trend_lines[line_index]
            (fit_type, _, popt, pcov) = trend_line
            fit_max = observation_day + i
            fit_min = observation_day + i - self.fit_points
            norm_x = fit_min
            norm_y = df[y_col].iloc[fit_min:fit_max].max()
            trend = [None] * len(df)
            if fit_type == "exp":
                for day in range(observation_day, len(df)):
                    trend[day] = self.exp_fit(
                        (day - norm_x + 1), *popt) * norm_y
            elif fit_type == "poly":
                for day in range(observation_day, len(df)):
                    trend[day] = self.poly_fit(
                        (day - norm_x + 1), *popt) * norm_y
            df[trend_cols[i]] = trend
        return df


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
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="log debug messages")
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
    logger.info("Starting...")
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("Logging debug messages...")
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
