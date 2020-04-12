#!/usr/bin/python

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ImageMagickFileWriter
from matplotlib.animation import PillowWriter
import seaborn as sns
import pandas as pd
import requests
import numpy as np
import datetime
import os.path
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
# For ImageMagick configuration, see https://stackoverflow.com/questions/23417487/saving-a-matplotlib-animation-with-imagemagick-and-without-ffmpeg-or-mencoder/42565258#42565258
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
    url = "https://docs.google.com/spreadsheets/d/1D6okqtBS3S2NRC7GFVHzaZ67DuTw7LX49-fqSLwJyeo/export?format=xlsx"
    update_time = datetime.date.fromtimestamp(os.path.getmtime("canada.xlsx"))
    if fetch or datetime.date.today() != update_time:
        print("Fetching spreadsheet")
        r = requests.get(url)
        with open("canada.xlsx", "wb") as output:
            output.write(r.content)
    df = pd.read_excel("canada.xlsx", sheet_name = "Cases", skiprows = range(0, 3), index_col=0)
    df_agg = df["date_report"].to_frame().groupby("date_report").apply(lambda x: x.count())
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
def exp_fit(x, a, k, b):
    """ 
    Fitting function for curve fitting (below)
    """
    return a * np.exp(x * k) + b


def fit_trends(df, start_day, observation_day, fit_points=FIT_POINTS):
    """
    Fit a line through the observations in the data frame
    """
    #df = data[data["day"].between(start_day, observation_day, inclusive=True)].copy()
    x = df["day"].iloc[observation_day - fit_points: observation_day + 1]
    y = df["cumulative"].iloc[observation_day - fit_points: observation_day + 1]
    norm_x = x.min()
    norm_y = y.max()
    x_normalized = x - norm_x + 1
    y_normalized = y/norm_y
    #print(x, y, x_normalized, y_normalized)
    # feed it into scipy, method is one of ‘lm’, ‘trf’, ‘dogbox’}
    popt, pcov = curve_fit(exp_fit, x_normalized, y_normalized, 
                           method="trf", p0=(3, 0.01, -6), maxfev = 10000)
    
    # add columns to the data frame holding trend values to the dataframe
    series = "fit_{}".format(observation_day)
    df[series] = x_normalized.apply(lambda x: int(exp_fit(x, *popt) * norm_y))
    return(df, popt)


def extrapolate_trends(df, popt, 
                       start_day, 
                       observation_day,
                       fit_points=FIT_POINTS,
                       days_extrapolation=DAYS_EXTRAPOLATION):
    norm_x = df["day"].loc[observation_day - fit_points:].min()
    norm_y = df["cumulative"].loc[observation_day - fit_points:observation_day].max()
    # print(observation_day, norm_x, norm_y)
    x_normalized = df["day"] - norm_x + 1
    y_normalized = df["cumulative"]/norm_y
    
    observation_row = df.loc[observation_day]
    for days in range(1,days_extrapolation):
        this_days = observation_row["day"] + days
        this_date = observation_row["date"] + datetime.timedelta(days=days)
        df.loc[this_days, "fit_{}".format(observation_day)] = \
                int(exp_fit((this_days - norm_x + 1), *popt) * norm_y)
    return df

def double_time(df, start_date, observation_date):
    obs_day = df.loc[observation_date, "day"]
    y = 0.5 * df.loc[observation_date, "cumulative"]
    x0 = df[df["cumulative"] <= y]["day"].max()
    x1 = df[df["cumulative"] >= y]["day"].min()
    y0 = df[df["cumulative"] <= y]["cumulative"].max()
    y1 = df[df["cumulative"] >= y]["cumulative"].min()
    x = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    double_days = obs_day - x
    print(x0, x1, x, y0, y1, y, obs_day, double_days)
    return double_days


def multifit(df, start_day, most_current_day, fit_points, frame_count):
    for frame in range(frame_count): 
        observation_day = most_current_day - frame_count + frame + 1
        days_extrapolation = most_current_day - observation_day
        (df, popt) = fit_trends(df, start_day=start_day, 
                                observation_day=observation_day, 
                                fit_points=fit_points)
        df = extrapolate_trends(df, popt, start_day=start_day,
                                observation_day=observation_day,
                                fit_points=fit_points,
                                days_extrapolation=days_extrapolation)
        print("Prediction {}: {}".format(frame,
                                         int(df['fit_{}'.format(observation_day)].max())))
        #print(df['fit_{}'.format(observation_day)])
    return df


def next_frame(i, text, lines, start_day, most_current_day, 
               fit_points, frame_count, df, ax):
    base_frame = int(i/INTERPOLATIONS)
    interpolate = (i % INTERPOLATIONS) / INTERPOLATIONS
    observation_day = most_current_day - frame_count + base_frame + 1
    days_extrapolation = most_current_day - observation_day
    if interpolate != 0 and observation_day < most_current_day: # interpolate
        yfit_lower = df[df["day"] >=
                        start_day]["fit_{}".format(observation_day)].to_list()
        yfit_upper = df[df["day"] >=
                        start_day]["fit_{}".format(observation_day + 1)].to_list()
        yfit = [yfit_lower[j] + interpolate * (yfit_upper[j] - yfit_lower[j]) for j in
                range(len(yfit_lower))]
        #print(yfit)
    else:
        yfit = df[df["day"] >=
                  start_day]["fit_{}".format(observation_day)].to_list()
    text_value = round(int(df["fit_{}".format(observation_day)].max()), -3)
    text.set_text("{:d}".format(text_value))
    text.set_text("")
    text.set_position((df["day"].max() + 0.5, min(YLIM_MAX,
                                                   df["fit_{}".format(observation_day)].max())))
    yobs = df[df["day"] >= start_day]["cumulative"].to_list()
    #TODO None out the entries for day > observation_day
    yobs = [yobs[i] if i < (observation_day-start_day) else None for i in
            range(len(yobs))]
    #yfit = df[df["day"] >= start_day]["fit_{}".format(observation_day)]
    lines[0].set_ydata(yobs)
    lines[1].set_ydata(yfit)
    #ax.set_title("Observation day {}".format(observation_day))
    ax.set_title("Flattening the curve in Canada")
    # print(label, df.index.max(), df["day"].max(), df["fitted"].max())
    return (ax, text)


def main(args):
    df0 = get_df()
    most_current_day = df0["day"].max()
    start_day = df0["day"].min() + START_DAYS_OFFSET
    observation_day = most_current_day
    fit_points=FIT_POINTS
    frame_count = FRAME_COUNT
    df1 = multifit(df0, start_day, most_current_day, fit_points, frame_count)
    df1 = df1[df1["day"] >= start_day].copy() 

    # initial plot
    fig = plt.figure()
    plt.xlabel("Date")
    plt.ylabel("Cumulative cases")
    plt.title("Covid-19 extrapolations")
    ax = plt.gca()
    xleft = df1["day"].min()
    xright = df1["day"].max()
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(bottom=0, top=YLIM_MAX)
    lines = ax.plot(df1["day"], df1["cumulative"], "ro",
                    df1["day"], df1["cumulative"], "b-", markersize=8)
    text = plt.text(x=df1["day"].max() + 0.2, y=df1["cumulative"].max(), s="", 
                    ha="left") 
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=7, offset=0))
    xlabels = [df1.iloc[i]["date"].strftime("%b %d") for i in range(0, len(df1), 7)]
    ax.set_xticklabels(xlabels)
    fargs = [text, lines, start_day, most_current_day, 
             fit_points, frame_count, df1, ax]
    anim = FuncAnimation(fig, next_frame, frames=np.arange(INTERPOLATIONS * frame_count), 
                         fargs=fargs, interval=100,
                         repeat=True, repeat_delay=2000)
    if args.save:
        writer = ImageMagickFileWriter()
        # writer = PillowWriter(fps=2)
        anim.save('covid.gif', writer=writer)
    else:
        plt.show()
    for days_in_past in range(10,-1,-5):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description = "Animate some Covid data.",
                                    fromfile_prefix_chars = '@' )
    # TODO Specify your real parameters here.
    parser.add_argument("--save",
                        action="store_true",
                        help = "save the gif file instead of displaying it",
                        default=False)
    args = parser.parse_args()
    main(args)
