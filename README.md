# Animation of Covid data for Canada

The python script in this project generates the GIF files below. It is an exercise done out of interest, not intended for serious use.

The idea is to show how predictions of today's numbers (at the far right) have changed over time, converging on the actual value at the end of the animation.

Each line is a best fit to the previous several days of cumulative Covid-19 cases in Canada, using an exponential curve. In early March, the predictions suggested a big growth into the future, so that by now (mid April) some predictions would give us over a hundred thousand cases. Over time, those predictions have come down.

The fitting is naive. This is not sophisticated modeling and I am not an epidemiologist. The purpose is to illustrate what I hope is obvious, that the unprecedented physical distancing measures that we have all been taking over the past several weeks have had an effect. That things could have been so much worse.

The data comes from the COVID-19 Canada Open Data Working Group. Epidemiological Data from the COVID-19 Outbreak in Canada. https://github.com/ishaberry/Covid19Canada. 


## Details

To save a GIF file the script uses ImageMagick. Configuration for Windows is described in the top answer to [this Stack Overflow question](https://stackoverflow.com/questions/23417487/aving-a-matplotlib-animation-with-imagemagick-and-without-ffmpeg-or-mencoder/42565258#42565258). In summary:

- Install ImageMagick. I use the [chocolatey](https://chocolatey.org/) package manager for Windows and installed ImageMagick with that.
- Set an environment variable MAGICK_HOME pointing to the '<your-install-dir>\modules\coders' folder. You will need to start a new command prompt session after setting this for your environment to pick it up.
- Point MatPlotLib to the install directory. This is hard-coded for now. I may move it to a config file is there is any interest. It's the IMAGEMAGICK_DIR constant.

This post [How to create animated graphs in python](https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1) by Viviane Kakerbeck was very helpful. 

## Usage

To display an animated graph of cumulative cases (the default):

> python animate_covid.py
or
> python animate_covid.py -p cases

To display an animated graph of growth rate:

> python animate_covid.py -p growth

To save the graph in a gif or mp4 file in the current directory instead of displaying it

> python animate_covid.py -p <plot_type> -s [gif] | mp4

## Bar Chart races

There are several blog posts about implementing bar chart races in matplotlib.

- A basic outline is by [Pratap Vardhan](https://towardsdatascience.com/bar-chart-race-in-python-with-matplotlib-8e687a5c8a41)
- Slightly smoother, from [Data Science Coffee](https://www.datasciencecoffee.com/2020-smooth-bar-chart-race/)
- The most advanced, with properly smooth transitions, is by [Ted Petrou](https://medium.com/dunder-data/create-a-bar-chart-race-animation-in-python-with-matplotlib-477ed1590096) on May 4 (last week!). It is accompanied by a package put on PyPi. I use this package.

## Output

<video width="640" height="480" autoplay controls>
  <source src="covid_cases.mp4" type="video/mp4">
  <source src="movie.ogg" type="video/ogg">
  Your browser does not support the video tag.
</video>

![](covid_cases.gif)

![](covid_growth.gif)
