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
# Set up graphics
#-------------------------------------------------------------------------------
plt.style.use("ggplot")
mpl.rcParams['figure.figsize'] = [8.0, 5.0]
mpl.rcParams['figure.dpi'] = 90
mpl.rcParams['savefig.dpi'] = 100
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['legend.fontsize'] = 'large'
# mpl.rcParams['figure.titlesize'] = 'medium'
# For ImageMagick configuration, see
# https://stackoverflow.com/questions/23417487/saving-a-matplotlib-animation-with-imagemagick-and-without-ffmpeg-or-mencoder/42565258#42565258
plt.rcParams['animation.convert_path'] = "/Program Files/ImageMagick-7.0.10-Q16/magick.exe"
sns.set()

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
DAYS_EXTRAPOLATION = 14
DAYS_ANNOTATE = 6
START_DAYS_OFFSET = 15
FIT_POINTS = 7
FETCH_ALWAYS = False
FRAME_COUNT = 30
YLIM_MAX = 30000
INTERPOLATIONS = 5

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
    df_agg = df_download["date_report"].to_frame().groupby("date_report").apply(lambda x: x.count())
    # df_agg has an index named date_report and a single column, named
    # date_report. Fix this up to give a default index and a column "new_cases"
    df_agg.rename(columns={"date_report": "new_cases"}, inplace=True)
    df_agg.reset_index(inplace=True)
    df_agg.rename(columns={"date_report": "date"}, inplace=True)
    df_agg["date"] = [x.date() for x in df_agg["date"]]
    df_agg["day"] = df_agg.index
    df_agg["cumulative"] = df_agg["new_cases"].to_frame().expanding(1).sum()
    df_agg["cumulative_shift1"] = df_agg["cumulative"].shift(1)
    df_agg["cumulative_shift2"] = df_agg["cumulative"].shift(2)
    df_agg["three_day_moving_average"] = (df_agg["cumulative"] +
                                          df_agg["cumulative_shift1"] +
                                          df_agg["cumulative_shift2"]) / 3.0
    df_agg["cumulative"] = df_agg["three_day_moving_average"]
    df_agg["growth_rate"] = 100 * (df_agg["new_cases"] /
                                   df_agg["cumulative_shift1"])
    return df_agg

# curve fit: define the fitting function
def exp_fit(x_var, a_factor, k_exponent, b_intercept):
    """
    Fitting function for curve fitting (below)
    """
    return a_factor * np.exp(x_var * k_exponent) + b_intercept


def fit_trends(data, observation_day, fit_points=FIT_POINTS):
    """
    Fit a line through the observations in the data frame
    """
    x_list = data["day"].iloc[observation_day - fit_points: observation_day + 1]
    y_list = data["cumulative"].iloc[observation_day - fit_points: observation_day + 1]
    norm_x = x_list.min()
    norm_y = y_list.max()
    x_normalized = x_list - norm_x + 1
    y_normalized = y_list/norm_y
    #print(x, y, x_normalized, y_normalized)
    # feed it into scipy, method is one of ‘lm’, ‘trf’, ‘dogbox’}
    popt, pcov = curve_fit(exp_fit, x_normalized, y_normalized,
                           method="trf", p0=(3, 0.01, -6), maxfev=10000)
    # add columns to the data frame holding trend values to the dataframe
    series = "fit_{}".format(observation_day)
    data[series] = x_normalized.apply(lambda x: int(exp_fit(x, *popt) * norm_y))
    return(data, popt)


def extrapolate_trends(data, popt,
                       observation_day,
                       fit_points=FIT_POINTS,
                       days_extrapolation=DAYS_EXTRAPOLATION):
    """
    Given a fitted function, use it to extrapolate the data for
    a given observation day into the future (for that day).
    """
    norm_x = data["day"].loc[observation_day - fit_points:].min()
    norm_y = data["cumulative"].loc[observation_day - fit_points:observation_day].max()
    observation_row = data.loc[observation_day]
    for days in range(1, days_extrapolation):
        this_days = observation_row["day"] + days
        data.loc[this_days, "fit_{}".format(observation_day)] = \
                int(exp_fit((this_days - norm_x + 1), *popt) * norm_y)
    return data

def double_time(data, observation_date):
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


def multifit(data, most_current_day, fit_points, frame_count):
    """
    Call fit_trends and extrapolate_trends for each day, adding
    a column for each day's trend to the dataframe.
    """
    for frame in range(frame_count):
        observation_day = most_current_day - frame_count + frame + 1
        days_extrapolation = most_current_day - observation_day
        (data, popt) = fit_trends(data,
                                  observation_day=observation_day,
                                  fit_points=fit_points)
        data = extrapolate_trends(data, popt,
                                  observation_day=observation_day,
                                  fit_points=fit_points,
                                  days_extrapolation=days_extrapolation)
        print("Prediction {}: {}".format(frame,
                                         int(data['fit_{}'.format(observation_day)].max())))
    return data


def next_frame(i, artists, most_current_day, frame_count, data):
    """
    Function called from animator to generate frame i of the animation.
    """
    (text, lines, axis) = artists
    base_frame = int(i/INTERPOLATIONS)
    interpolate = (i % INTERPOLATIONS) / INTERPOLATIONS
    observation_day = most_current_day - frame_count + base_frame + 1
    if interpolate != 0 and observation_day < most_current_day: # interpolate
        yfit_lower = data["fit_{}".format(observation_day)].to_list()
        yfit_upper = data["fit_{}".format(observation_day + 1)].to_list()
        yfit = [yfit_lower[j] + interpolate * (yfit_upper[j] - yfit_lower[j]) for j in
                range(len(yfit_lower))]
    else:
        yfit = data["fit_{}".format(observation_day)].to_list()
    text_value = round(int(data["fit_{}".format(observation_day)].max()), -3)
    text.set_text("{:d}".format(text_value))
    text.set_text("")
    text.set_position(
        (data["day"].max() + 0.5, min(
            YLIM_MAX, data["fit_{}".format(observation_day)].max())))
    yobs = data["cumulative"].to_list()
    # Set yobs to None for day > observation_day
    yobs = [yobs[i] if i < (observation_day-data["day"].min()) else None for i in
            range(len(yobs))]
    lines[0].set_ydata(yobs)
    lines[1].set_ydata(yfit)
    axis.set_title("Flattening the curve in Canada")
    return (axis, text)


def main():
    """
    Entry point.
    """
    args = parse_args()
    df0 = get_df()
    most_current_day = df0["day"].max()
    start_day = df0["day"].min() + START_DAYS_OFFSET
    fit_points = FIT_POINTS
    frame_count = FRAME_COUNT
    df1 = multifit(df0, most_current_day, fit_points, frame_count)
    df1 = df1[df1["day"] >= start_day].copy()

    # initial plot
    fig = plt.figure()
    plt.xlabel("Date")
    plt.ylabel("Cumulative cases")
    plt.title("Covid-19 extrapolations")
    axis = plt.gca()
    axis.set_xlim(left=df1["day"].min(), right=df1["day"].max())
    axis.set_ylim(bottom=0, top=YLIM_MAX)
    lines = axis.plot(df1["day"], df1["cumulative"], "ro",
                      df1["day"], df1["cumulative"], "b-", markersize=8)
    text = plt.text(x=df1["day"].max() + 0.2, y=df1["cumulative"].max(), s="",
                    ha="left")
    axis.xaxis.set_major_locator(ticker.IndexLocator(base=7, offset=0))
    xlabels = [df1.iloc[i]["date"].strftime("%b %d") for i in range(0, len(df1), 7)]
    axis.set_xticklabels(xlabels)
    artists = (text, lines, axis)
    fargs = [artists, most_current_day, frame_count, df1]
    anim = FuncAnimation(fig, next_frame, frames=np.arange(INTERPOLATIONS * frame_count),
                         fargs=fargs, interval=100,
                         repeat=True, repeat_delay=2000)
    if args.save:
        writer = ImageMagickFileWriter()
        anim.save('covid.gif', writer=writer)
    else:
        plt.show()


def parse_args():
    """
    Define, read and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Animate some Covid-19 data.",
        usage="%(prog)s [options]",
        fromfile_prefix_chars='@')
    parser.add_argument("--save",
                        action="store_true",
                        help="save the gif file instead of displaying it",
                        default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
