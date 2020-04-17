#!/usr/bin/python3
"""
Module to generate animations of estimates and actual over time
looking backwards from today.

The data comes from the COVID-19 Canada Open Data Working Group.
Epidemiological Data from the COVID-19 Outbreak in Canada.
https://github.com/ishaberry/Covid19Canada.
"""

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import argparse
import os.path
import datetime
# import configparser
# import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ImageMagickFileWriter
import seaborn as sns
import pandas as pd
import requests
from scipy.optimize import curve_fit

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
DAYS_EXTRAPOLATION = 14
DAYS_ANNOTATE = 6
START_DAYS_OFFSET = 10
FIT_POINTS = 7
FETCH_ALWAYS = False
FRAME_COUNT = 32
YSCALE = "linear"  # "linear", "log", "symlog", "logit",
INTERPOLATIONS = 5
# TODO: IMAGEMAGICK_EXE is hardcoded here. Put it in a config file.
# IMAGEMAGICK_EXE = "/Program Files/ImageMagick-7.0.9-Q16/magick.exe"
IMAGEMAGICK_EXE = "/Program Files/ImageMagick-7.0.10-Q16/magick.exe"
# For ImageMagick configuration, see
# https://stackoverflow.com/questions/23417487/saving-a-matplotlib-animation-with-imagemagick-and-without-ffmpeg-or-mencoder/42565258#42565258

#-------------------------------------------------------------------------------
# Set up graphics
#-------------------------------------------------------------------------------
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [7.0, 4.0]
mpl.rcParams['figure.dpi'] = 90
mpl.rcParams['savefig.dpi'] = 100
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['legend.fontsize'] = 'large'
# mpl.rcParams['figure.titlesize'] = 'medium'
sns.set()
sns.set_palette("muted")
plt.rcParams['animation.convert_path'] = IMAGEMAGICK_EXE


#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_df(fetch=FETCH_ALWAYS):
    """
    Download data from the COVID-19 Canada Open Data Working Group.
    Only download once a day (unless FETCH_ALWAYS is set) to avoid
    unnecessary refetching of the same data.
    """
    url = ("https://docs.google.com/spreadsheets/d/"
           "1D6okqtBS3S2NRC7GFVHzaZ67DuTw7LX49-fqSLwJyeo/export?format=xlsx")
    update_time = datetime.date.fromtimestamp(os.path.getmtime("canada.xlsx"))
    if fetch or datetime.date.today() != update_time:
        print("Fetching spreadsheet")
        response = requests.get(url)
        with open("canada.xlsx", "wb") as output:
            output.write(response.content)
    df_download = pd.read_excel("canada.xlsx",
                                sheet_name="Cases",
                                skiprows=range(0, 3),
                                index_col=0)
    return df_download


class Plot():
    """
    Generic Plot class.
    There's nothing here yet, but it will probably fill up as more plots
    are added
    """


class GrowthRate(Plot):
    """
    Plot the percentage change in daily new cases
    """
    def __init__(self, data, save_output=False):
        """
        Initialize class variables and call what needs to be called.
        """
        self.data = data
        self.save_output = save_output
        self.prep()
        growth_rate_list = self.smooth("growth_rate")
        growth_rate_list = self.interpolate(growth_rate_list)
        self.plot(growth_rate_list)

    def prep(self):
        """
        Take the raw dataframe and aggregate it so it can be used by this plot.
        """
        self.data = self.data["date_report"].to_frame().groupby(
            "date_report").apply(lambda x: x.count())
        self.data.rename(columns={"date_report": "new_cases"}, inplace=True)
        self.data.reset_index(inplace=True)
        self.data.rename(columns={"date_report": "date"}, inplace=True)
        self.data["date"] = [x.date() for x in self.data["date"]]
        self.data["day"] = self.data.index
        self.data["cumulative"] = self.data["new_cases"].to_frame().expanding(
            1).sum()
        self.data["cumulative_shift1"] = self.data["cumulative"].shift(1)
        self.data["growth_rate"] = 100 * (self.data["new_cases"] /
                                          self.data["cumulative_shift1"])

    def smooth(self, column, degree=3):
        """
        Smooth the values in self.data[column], returning a list,
        Taken from
        https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
        """
        unsmoothed = self.data[column].to_list()
        window = degree * 2 - 1
        weight = np.array([1.0] * window)
        weight_gauss = []
        for i in range(window):
            i = i - degree + 1
            frac = i / float(window)
            gauss = 1 / (np.exp((4 * frac)**2))
            weight_gauss.append(gauss)
        weight = np.array(weight_gauss) * weight
        smoothed = [0.0] * len(unsmoothed)
        for i, _ in enumerate(smoothed):
            if i <= len(unsmoothed) - window:
                smoothed[i] = sum(
                    np.array(unsmoothed[i:i + window]) * weight) / sum(weight)
            else:
                smoothed[i] = float("NaN")
        return smoothed

    def interpolate(self, growth_rate):
        """
        Interpolate INTERPOLATIONS between every pair in growth_rate
        """
        new_length = (len(growth_rate) - 1) * INTERPOLATIONS
        interpolated = [0.0] * new_length
        for i in range(new_length):
            quotient = int(i / INTERPOLATIONS)
            mu_1 = (i % INTERPOLATIONS) / INTERPOLATIONS
            # use cosine interpolation for smoother than linear interpolation
            mu_2 = (1 - np.cos(mu_1 * np.pi)) / 2
            interpolated[i] = ((1 - mu_2) * growth_rate[quotient] +
                               mu_2 * growth_rate[quotient + 1])
        return interpolated

    def plot(self, growth_rate):
        """
        Plot it
        """
        days = [i / INTERPOLATIONS for i in range(len(growth_rate))]
        start_day = self.data["day"].min() + START_DAYS_OFFSET
        df1 = self.data[self.data["day"] >= start_day].copy()
        # Initial plot
        fig = plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Percentage increase")
        plt.title("Covid-19 daily percentage increase in cases")
        plt.yscale(YSCALE)
        axis = plt.gca()
        axis.plot(days, growth_rate, "-", lw=3, alpha=0.8)
        axis.xaxis.set_major_locator(ticker.IndexLocator(base=7, offset=0))
        axis.set_xlim(left=min(days), right=max(days) - 3)
        if YSCALE == "log":
            axis.set_ylim(bottom=1, top=1.1 * np.nanmax(growth_rate))
        else:
            axis.set_ylim(bottom=0, top=1.1 * np.nanmax(growth_rate))
        xlabels = [
            df1.iloc[i]["date"].strftime("%b %d")
            for i in range(0, len(df1), 7)
        ]
        axis.set_xticklabels(xlabels)
        anim = FuncAnimation(fig,
                             self.next_frame,
                             frames=np.arange(len(days)),
                             fargs=[axis, growth_rate],
                             interval=50,
                             repeat=True,
                             repeat_delay=1500)
        if self.save_output:
            writer = ImageMagickFileWriter()
            anim.save('covid_growth.gif', writer=writer)
        else:
            plt.show()

    def next_frame(self, i, axis, growth_rate):
        """
        Function called from animator to generate frame i of the animation
        """
        y_values = growth_rate.copy()
        for j, _ in enumerate(y_values):
            if j > i:
                y_values[j] = None
        axis.get_lines()[0].set_ydata(y_values)
        return axis, growth_rate


class CumulativeCases(Plot):
    """
    Plot cumulative cases
    """
    def __init__(self,
                 data,
                 save_output=False,
                 frame_count=FRAME_COUNT,
                 fit_points=FIT_POINTS):
        """
        Initialize the class variables and call what needs to be called.
        The dataframe "data" has a row for each case.
        It must have the following columns:
        - "date_report": the date a case is reported
        """
        self.data = data
        self.save_output = save_output
        self.frame_count = frame_count
        self.fit_points = fit_points
        self.prep()
        self.smooth()
        self.plot()

    def prep(self):
        """
        Take the raw dataframe and aggregate it so it can be used by this plot.
        """
        self.data = self.data["date_report"].to_frame().groupby(
            "date_report").apply(lambda x: x.count())
        # df_agg has an index named date_report and a single column, named
        # date_report. Fix this up to give a default index and a column "new_cases"
        self.data.rename(columns={"date_report": "new_cases"}, inplace=True)
        self.data.reset_index(inplace=True)
        self.data.rename(columns={"date_report": "date"}, inplace=True)
        self.data["date"] = [x.date() for x in self.data["date"]]
        self.data["day"] = self.data.index
        self.data["cumulative"] = self.data["new_cases"].to_frame().expanding(
            1).sum()

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
        plt.title("Flattening the curve in Canada")
        axis = plt.gca()
        plt.yscale(YSCALE)
        axis.set_xlim(left=df1["day"].min(), right=df1["day"].max())
        if YSCALE == "log":
            axis.set_ylim(bottom=10, top=df1["cumulative"].max() * 1.5)
        else:
            axis.set_ylim(bottom=0, top=df1["cumulative"].max() * 1.5)
        axis.plot(df1["day"], df1["cumulative"], "o", markersize=8, alpha=0.8)
        axis.plot(df1["day"], df1["cumulative"], "-", lw=2, alpha=0.8)
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
        if self.save_output:
            writer = ImageMagickFileWriter()
            anim.save('covid.gif', writer=writer)
        else:
            plt.show()

    def exp_fit(self, x_var, a_factor, k_exponent, b_intercept):
        """
        Fitting function for curve fitting (below)
        """
        return a_factor * np.exp(x_var * k_exponent) + b_intercept

    def fit_trends(self, popt, observation_day):
        """
        Fit a line through the observations in the data frame
        """
        x_list = self.data["day"].iloc[observation_day -
                                       self.fit_points:observation_day + 1]
        y_list = self.data["cumulative"].iloc[observation_day -
                                              self.fit_points:observation_day +
                                              1]
        norm_x = x_list.min()
        norm_y = y_list.max()
        x_normalized = x_list - norm_x + 1
        y_normalized = y_list / norm_y
        #print(x, y, x_normalized, y_normalized)
        # feed it into scipy, method is one of ‘lm’, ‘trf’, ‘dogbox’}
        popt, pcov = curve_fit(self.exp_fit,
                               x_normalized,
                               y_normalized,
                               method="trf",
                               p0=popt,
                               maxfev=10000)
        return popt, pcov

    def add_series_from_fit(self, popt, observation_day):
        """
        Given a fitted function for a given observation day,
        use it to add a column to the dataframe "data" with
        values of that function for the whole data range.
        """
        norm_x = self.data["day"].loc[observation_day - self.fit_points:].min()
        norm_y = self.data["cumulative"].loc[observation_day - self.
                                             fit_points:observation_day].max()
        for day in range(len(self.data)):
            self.data.loc[day, "fit_{}".format(observation_day)] = \
                    int(self.exp_fit((day - norm_x + 1), *popt) * norm_y)

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
            popt, pcov = self.fit_trends(popt, observation_day=observation_day)
            self.add_series_from_fit(popt, observation_day=observation_day)
            print("Prediction {}: {}".format(
                frame, int(self.data['fit_{}'.format(observation_day)].max())))

    def next_frame(self, i, axis, most_current_day, data):
        """
        Function called from animator to generate frame i of the animation.
        """
        #(texts, lines, axis) = artists
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
        if interpolate != 0 and observation_day < most_current_day:  # interpolate
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
        # Set yobs to None for day > observation_day
        yobs = [
            yobs[i] if i <= (observation_day - data["day"].min()) else None
            for i in range(len(yobs))
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
        "--save",
        action="store_true",
        default=False,
        help="save the gif file instead of displaying it",
    )
    parser.add_argument("-p",
                        "--plot",
                        metavar="plot",
                        action="store",
                        type=str,
                        default="cases",
                        help="save the gif file instead of displaying it")
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
    args = parse_args()
    config = read_config(args)
    data = get_df()
    if args.plot == "cases":
        CumulativeCases(data, args.save)
    elif args.plot == "growth":
        GrowthRate(data, args.save)
    else:
        print("Unknown plot choice: {}".format(args.plot))


if __name__ == '__main__':
    main()
