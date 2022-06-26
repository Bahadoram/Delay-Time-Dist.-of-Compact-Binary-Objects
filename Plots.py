import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import time

from scipy.optimize      import curve_fit
from scipy               import stats
from matplotlib.gridspec import GridSpec

def power_law(x, a, b):
    '''
    Model for power law fit.

    Inputs:
        - x: independent variable
        - a: exponent
        - b: parameter to multiply by

    Output:
        - result of power law computation
    '''
    return x ** a * 10 ** b

def chisqr(obs, exp):
    '''
    Chisquare computation (Person's chi square)

    Inputs:
        - obs: array of observed values
        - exp: array of expected values

    Outputs:
        - value of chi square
        - p-value
    '''
    chisqr = 0
    for i in range(len(obs)):
        chisqr+=((obs[i]-exp[i])**2)/obs[i]
    return chisqr, stats.chi2.sf(chisqr, len(obs))

# t-student distribution
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))

def plot_complete(df, column, Delay=False, bins=100, save=False, filename=None):
    '''
    Plot the whole dataset.

    Inputs:
        - df: dataframe with the data
        - column: column you want to plot (usually Delay_Time or GWtime)
        - Delay: boolean, if true plot the sum of merger time and BWorldtime,
                 else plot just the merger time distribution
        - bins: number of bins in the histogram
        - save: boolean: if true save figure
        - filename: name of the file to be saved

    Outputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
    '''
    fig, ax = plt.subplots(figsize=(15,7))

    if Delay==False:
        BWorldtime = 0
    else:
        BWorldtime = df['BWorldtime']

    #b = np.histogram_bin_edges(BHBH[(BHBH.GWtime<1e18)].GWtime, bins='rice') # does not work with bind='fd'
    b = np.logspace(np.log10(min(df[column]+BWorldtime)), np.log10(max(df[column]+BWorldtime)), bins)
    entries, edges, _ = ax.hist(df[column]+BWorldtime, bins=b, density=True)

    # calculate bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title('Distribution of the Merger times')
    ax.set_xlabel('Delay Time [Myr]')
    ax.set_ylabel('PDF [log]')

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show()

    return bin_centers, entries

def fit_complete(bin_centers, entries, xmin=0, xmax=1e30, save=False, filename=None):
    '''
    Fit the distribution with a unique function.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xmin: value in which the fit should start
        - xmax: value in which the fit should end
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''

    fig, ax = plt.subplots(nrows=2, figsize=(15,12))

    mask = np.where((bin_centers>xmin) & (bin_centers<xmax) & (entries!=0))

    results = stats.linregress(np.log10(bin_centers[mask]), np.log10(entries[mask]))

    ax[0].scatter(np.log10(bin_centers[mask]), np.log10(entries[mask]), s=30, marker='x', label='Real data', color='black')
    ax[0].plot(np.log10(bin_centers[mask]), results.intercept + results.slope*np.log10(bin_centers[mask]), '--', lw=3, label='Power Law fit', color='red')

    ts = tinv(0.05, len(bin_centers[mask])-2)
    print(f"slope (95%):\t {results.slope:.6f} +/- {ts*results.stderr:.6f}")
    print(f"intercept (95%): {results.intercept:.6f}"
          f" +/- {ts*results.intercept_stderr:.6f}")

    chi2, p = stats.chisquare(results.intercept + results.slope*np.log10(bin_centers[mask]),
                                np.log10(entries[mask]), ddof=len(entries[mask]))

    print(f"chi2 statistic:\t{chi2:.5g}")
    print(f"p-value:       \t{p:.5g}\n")

    # get the covariance. Numpy conveniently provides the np.cov() method
    cov = np.cov(np.log10(bin_centers[mask]), np.log10(entries[mask]), ddof=2)
    b_hat = cov[0, 1] / cov[0, 0]
    a_hat = np.mean(np.log10(entries[mask]) - b_hat * np.log10(bin_centers[mask]))

    ssr = np.sum((np.log10(entries[mask]) - a_hat - b_hat * np.log10(bin_centers[mask]))**2)
    tss = np.sum((np.mean(np.log10(entries[mask])) - np.log10(entries[mask]))**2)

    rsq = 1 - ssr / tss

    print("R2 =", rsq, "\nR =", np.sqrt(rsq))
    print("Covariance matrix:\n", np.corrcoef(np.log10(bin_centers[mask]),np.log10(entries[mask]))) # check with the correlation matrix that R is the correlation coefficient
    print("\n\n")

    ax[0].set_xlabel('Delay Time [log/Myr]')
    ax[0].set_ylabel('PDF [log]')

    ax[0].grid(ls='dotted')
    ax[0].legend(loc='upper right')

    ax[1].scatter(np.log10(bin_centers[mask]), (results.intercept + results.slope*np.log10(bin_centers[mask])) - np.log10(entries[mask]), s=30, marker='x', label='Residuals', color='black')
    ax[1].axhline(y=0, color='r', linestyle='dashed', label="expected residual")

    ax[1].set_xlabel('Delay Time [log/Myr]')
    ax[1].set_ylabel('Residual [log]')

    ax[1].grid(ls='dotted')
    ax[1].legend(loc='upper right')

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show()


def fit_range(bin_centers, entries, xstep = (0, 1e7, 1e19, 1e30), save=False, filename=None):
    '''
    Fit the distribution with a function defined by steps.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xstep: list of steps
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''

    fig = plt.figure(constrained_layout=True, figsize=(15,12))

    gs  = GridSpec(2, 3, figure=fig)
    ax  = fig.add_subplot(gs[0, :])
    ax_res = (fig.add_subplot(gs[1, 0]),fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, -1]))

    ax.scatter(np.log10(bin_centers[np.where(entries!=0)]), np.log10(entries[np.where(entries!=0)]), s=50, marker='x', label='Real data', c='black')

    for i in range(len(xstep)-1):

        print("----- FIT", i, "-----\n")

        mask = np.where((bin_centers>xstep[i]) & (bin_centers<xstep[i+1]) & (entries!=0))
        results = stats.linregress(np.log10(bin_centers[mask]), np.log10(entries[mask]))
        ax.plot(np.log10(bin_centers[mask]), results.intercept + results.slope*np.log10(bin_centers[mask]), '--', lw=3, label='Range '+str(i))

        ts = tinv(0.05, len(bin_centers[mask])-2)
        print(f"slope (95%):\t {results.slope:.6f} +/- {ts*results.stderr:.6f}")
        print(f"intercept (95%): {results.intercept:.6f}"
          f" +/- {ts*results.intercept_stderr:.6f}")

        chi2, p = stats.chisquare(results.intercept + results.slope*np.log10(bin_centers[mask]),
                                  np.log10(entries[mask]), ddof=len(entries[mask]))

        print(f"chi2 statistic:\t{chi2:.5g}")
        print(f"p-value:       \t{p:.5g}\n")


        # get the covariance. Numpy conveniently provides the np.cov() method
        cov = np.cov(np.log10(bin_centers[mask]), np.log10(entries[mask]), ddof=2)
        b_hat = cov[0, 1] / cov[0, 0]
        a_hat = np.mean(np.log10(entries[mask]) - b_hat * np.log10(bin_centers[mask]))

        ssr = np.sum((np.log10(entries[mask]) - a_hat - b_hat * np.log10(bin_centers[mask]))**2)
        tss = np.sum((np.mean(np.log10(entries[mask])) - np.log10(entries[mask]))**2)

        rsq = 1 - ssr / tss

        print("R2 =", rsq, "\nR =", np.sqrt(rsq))
        print("Covariance matrix:\n", np.corrcoef(np.log10(bin_centers[mask]),np.log10(entries[mask]))) # check with the correlation matrix that R is the correlation coefficient
        print("\n\n")


        ax_res[i].scatter(np.log10(bin_centers[mask]), (results.intercept + results.slope*np.log10(bin_centers[mask])) - np.log10(entries[mask]), s=30, marker='x', color='black')
        ax_res[i].axhline(y=0, color='r', linestyle='dashed')

        ax_res[i].set_xlabel('Residual [log]')
        ax_res[i].set_ylabel('Delay Time [log/Myr]')

        ax_res[i].grid(ls='dotted')
        ax_res[i].legend(loc='upper right')


    ax.set_xlabel('Delay Time [Myr]')
    ax.set_ylabel('PDF [log]')

    ax.grid(ls='dotted')
    ax.legend(loc='upper right')

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show()

def Z_plot_figure(df, column, Delay=False, bins=100, save=False, filename=None):
    '''
    Plot the whole dataset divided by values of Z.

    Inputs:
        - df: dict of dataframes with the data
        - column: column you want to plot (usually Delay_Time or GWtime)
        - Delay: boolean, if true plot the sum of merger time and BWorldtime,
                 else plot just the merger time distribution
        - bins: number of bins in the histogram
        - save: boolean: if true save figure
        - filename: name of the file to be saved

    Outputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
    '''
    bin_centers, entries = (np.empty(shape=(3,5), dtype=object) for i in range(2))

    BWorldtime = {elem : pd.Series for elem in df}
    if Delay==False:
        title = 'Merger Time'
        for key in df.keys():
            BWorldtime[key] = 0
    else:
        title = 'Delay Time'
        for key in df.keys():
            BWorldtime[key] = df[key]['BWorldtime']

    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(30, 20))

    # indexes for the plot
    x, y = (0, 0)

    for key in df.keys():
        b                       = np.logspace(np.log10(min(df[key][column]+BWorldtime[key])), np.log10(max(df[key][column]+BWorldtime[key])), bins)
        entries[x,y], edges, _  = ax[x,y].hist(            df[key][column]+BWorldtime[key],   bins=b, density=True                           )

        # calculate bin centers
        bin_centers[x,y]          = 0.5 * (edges[:-1] + edges[1:])
        ax[x,y].set_xscale('log')
        ax[x,y].set_yscale('log')

        ax[x,y].set_title('Z = '+str(key))
        ax[x,y].set_xlabel('Delay Time [Myr]')
        ax[x,y].set_ylabel('PDF')

        y += 1
        if y == 5:
            x +=1
            y = 0

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

    return bin_centers, entries

def Z_fit_complete(bin_centers, entries, xmin=0, xmax=1e30, save=False, filename=None):
    '''
    Fit the distributions with a unique function.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xmin: value in which the fit should start
        - xmax: value in which the fit should end
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''
    results = np.empty(shape=(3,5), dtype=object)

    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(30, 20))

    keys = [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 
            0.001,   0.002,  0.004,  0.006,  0.008,
            0.01,    0.014,  0.017,   0.02,   0.03]

    # indexes for the plot
    x, y = (0, 0)

    for key in keys:
        ax[x,y].scatter(np.log10(bin_centers[x,y][entries[x,y]!=0]), np.log10(entries[x,y][entries[x,y]!=0]), s=30, marker='x', color='black')

        mask = np.where((bin_centers[x,y]>xmin) & (bin_centers[x,y]<xmax) & (entries[x,y]!=0))
        results[x,y] = stats.linregress(np.log10(bin_centers[x,y][mask]), np.log10(entries[x,y][mask]))

        ax[x,y].plot(np.log10(bin_centers[x,y][mask]), results[x,y].intercept + results[x,y].slope*np.log10(bin_centers[x,y][mask]), '--', lw=4)

        ax[x,y].set_title('Z = '+str(key))
        ax[x,y].set_xlabel('Delay Time [log/Myr]')
        ax[x,y].set_ylabel('PDF [log]')
        ax[x,y].grid(ls='dotted')

        y += 1
        if y == 5:
            x +=1
            y = 0

    labels=['PDF', "Fit"]
    fig.legend(labels, loc='center',bbox_to_anchor=(.5,.925), ncol=len(labels), bbox_transform=fig.transFigure)

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

def Z_fit_range(bin_centers, entries, xstep = (0, 1e7, 1e19, 1e30), save=False, filename=None):
    '''
    Fit the distribution with a function defined by steps.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xstep: list of steps
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''
    results = np.empty(shape=(3,5,3), dtype=object)

    keys = [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 
            0.001,   0.002,  0.004,  0.006,  0.008,
            0.01,    0.014,  0.017,   0.02,   0.03]
    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(30, 20))

    # indexes for the plot
    x, y = (0, 0)

    for key in keys:
        ax[x,y].scatter(np.log10(bin_centers[x,y][entries[x,y]!=0]), np.log10(entries[x,y][entries[x,y]!=0]), s=30, marker='x', color='black')

        mask = np.empty(shape=3, dtype=object)
        for i in range(len(results[x,y])):

            mask[i] = np.where((bin_centers[x,y]>xstep[i]) & (bin_centers[x,y]<xstep[i+1]) & (entries[x,y]!=0))
            results[x,y,i] = stats.linregress(np.log10(bin_centers[x,y][mask[i]]), np.log10(entries[x,y][mask[i]]))

            ax[x,y].plot(np.log10(bin_centers[x,y][mask[i]]), results[x,y,i].intercept + results[x,y,i].slope*np.log10(bin_centers[x,y][mask[i]]), '--', lw=4)

        ax[x,y].set_title('Z = '+str(key))
        ax[x,y].set_xlabel('Delay Time [Myr]')
        ax[x,y].set_ylabel('PDF [log]')
        ax[x,y].grid(ls='dotted')

        y += 1
        if y == 5:
            x +=1
            y = 0

    labels=['PDF', "Fit range 1", "Fit range 2", "Fit range 3"]
    fig.legend(labels, loc='center',bbox_to_anchor=(.5,.925), ncol=len(labels), bbox_transform=fig.transFigure)

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

def alpha_plot_figure(df, column, Delay=False, bins=100, save=False, filename=None):
    '''
    Plot the whole dataset divided by values of alpha.

    Inputs:
        - df: dict of dataframes with the data
        - column: column you want to plot (usually Delay_Time or GWtime)
        - Delay: boolean, if true plot the sum of merger time and BWorldtime,
                 else plot just the merger time distribution
        - bins: number of bins in the histogram
        - save: boolean: if true save figure
        - filename: name of the file to be saved

    Outputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
    '''
    bin_centers, entries = (np.empty(shape=(2,2), dtype=object) for i in range(2))

    BWorldtime = {elem : pd.Series for elem in df}
    if Delay==False:
        title = 'Merger Time'
        for key in df.keys():
            BWorldtime[key] = 0
    else:
        title = 'Delay Time'
        for key in df.keys():
            BWorldtime[key] = df[key]['BWorldtime']

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

    # indexes for the plot
    x, y = (0, 0)

    for key in df.keys():
        b                       = np.logspace(np.log10(min(df[key][column]+BWorldtime[key])), np.log10(max(df[key][column]+BWorldtime[key])), bins)
        entries[x,y], edges, _  = ax[x,y].hist(            df[key][column]+BWorldtime[key],   bins=b, density=True                           )

        # calculate bin centers
        bin_centers[x,y]          = 0.5 * (edges[:-1] + edges[1:])
        ax[x,y].set_xscale('log')
        ax[x,y].set_yscale('log')

        ax[x,y].set_title('α = '+str(key))
        ax[x,y].set_xlabel('Delay Time [Myr]')
        ax[x,y].set_ylabel('PDF')

        y += 1
        if y == 2:
            x +=1
            y = 0

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

    return bin_centers, entries

def alpha_fit_complete(bin_centers, entries, xmin=0, xmax=1e30, save=False, filename=None):
    '''
    Fit the distribution with a unique function divided by values of alpha.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xmin: value in which the fit should start
        - xmax: value in which the fit should end
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''
    results = np.empty(shape=(2,2), dtype=object)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

    keys = [0.5, 1.0, 3.0, 5.0]

    # indexes for the plot
    x, y = (0, 0)

    for key in keys:
        ax[x,y].scatter(np.log10(bin_centers[x,y][entries[x,y]!=0]), np.log10(entries[x,y][entries[x,y]!=0]), s=30, marker='x', color='black')

        mask = np.where((bin_centers[x,y]>xmin) & (bin_centers[x,y]<xmax) & (entries[x,y]!=0))
        results[x,y] = stats.linregress(np.log10(bin_centers[x,y][mask]), np.log10(entries[x,y][mask]))

        ax[x,y].plot(np.log10(bin_centers[x,y][mask]), results[x,y].intercept + results[x,y].slope*np.log10(bin_centers[x,y][mask]), '--', lw=4)

        ax[x,y].set_title('α = '+str(key))
        ax[x,y].set_xlabel('Delay Time [log/Myr]')
        ax[x,y].set_ylabel('PDF [log]')
        ax[x,y].grid(ls='dotted')

        y += 1
        if y == 2:
            x +=1
            y = 0

    labels=['PDF', "Fit"]
    fig.legend(labels, loc='center',bbox_to_anchor=(.5,.925), ncol=len(labels), bbox_transform=fig.transFigure)

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

def alpha_fit_range(bin_centers, entries, xstep = (0, 1e7, 1e19, 1e30), save=False, filename=None):
    '''
    Fit the distribution with a function defined by steps.

    Inputs:
        - bin_centers: center of each bin
        - entries: value of the pdf in each bin center
        - xstep: list of steps
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''
    results = np.empty(shape=(2,2,3), dtype=object)

    keys = [0.5, 1.0, 3.0, 5.0]
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

    # indexes for the plot
    x, y = (0, 0)

    for key in keys:
        ax[x,y].scatter(np.log10(bin_centers[x,y][entries[x,y]!=0]), np.log10(entries[x,y][entries[x,y]!=0]), s=30, marker='x', color='black')

        mask = np.empty(shape=3, dtype=object)
        for i in range(len(results[x,y])):

            mask[i] = np.where((bin_centers[x,y]>xstep[i]) & (bin_centers[x,y]<xstep[i+1]) & (entries[x,y]!=0))
            results[x,y,i] = stats.linregress(np.log10(bin_centers[x,y][mask[i]]), np.log10(entries[x,y][mask[i]]))

            ax[x,y].plot(np.log10(bin_centers[x,y][mask[i]]), results[x,y,i].intercept + results[x,y,i].slope*np.log10(bin_centers[x,y][mask[i]]), '--', lw=4)

        ax[x,y].set_title('α = '+str(key))
        ax[x,y].set_xlabel('Delay Time [Myr]')
        ax[x,y].set_ylabel('PDF [log]')
        ax[x,y].grid(ls='dotted')

        y += 1
        if y == 2:
            x +=1
            y = 0

    labels=['PDF', "Fit range 1", "Fit range 2", "Fit range 3"]
    fig.legend(labels, loc='center',bbox_to_anchor=(.5,.925), ncol=len(labels), bbox_transform=fig.transFigure)

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show

def plot_all(df, column, Delay=False, bins=100, obj='Z', save=False, filename=None):
    '''
    Plot all the distributions together.

    Inputs:
        - df: dict of dataframes with data
        - column: column to plot the distribution of
        - Delay: False: Plot merger time, true plot delay time
        - bins: number of bins
        - obj: 'Z' if metallicity, 'alpha' if common envelope
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''

    colors = plt.cm.rainbow(np.linspace(0, 1, len(df.keys())))

    BWorldtime = {elem : pd.Series for elem in df}
    if Delay==False:
        title = 'Merger Time'
        for key in df.keys():
            BWorldtime[key] = 0
    else:
        title = 'Delay Time'
        for key in df.keys():
            BWorldtime[key] = df[key]['BWorldtime']

    fig, ax = plt.subplots(figsize=(18,6))

    for key, c in zip(df.keys(), colors):
        b                    = np.logspace(np.log10(min(df[key][column]+BWorldtime[key])), np.log10(max(df[key][column]+BWorldtime[key])), bins )
        if obj=='Z':
            entries, edges, _    = ax.hist(df[key][column]+BWorldtime[key], bins=b, histtype='step', lw=3, label='Z = '+str(key), density=True, color=c )
        elif obj=='alpha':
            entries, edges, _    = ax.hist(df[key][column]+BWorldtime[key], bins=b, histtype='step', lw=3, label='α = '+str(key), density=True, color=c )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel('Delay Time [Myr]')
    ax.set_ylabel('PDF')
    ax.grid(ls='dotted')
    if obj=='Z':
        ax.legend(loc='upper right', ncol=3)
    elif obj=='alpha':
        ax.legend(loc='upper right', ncol=2)

    if save==True:
        fig.savefig('figures/'+filename)

    plt.show


def plot_hist(Y, nbins, log=False, save=False, filename=None):
    '''
    Plot the distribution of Y.

    Inputs:
        - Y: array to plot the distribution of
        - nbins: number of bins to use
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''
    fig, ax = plt.subplots(figsize=(15,7))

    if log==True:
        b = np.histogram_bin_edges(Y, bins='auto')
        ax.set_xlabel('Delay Time [log/Myr]')
        ax.set_ylabel('PDF [log]')

    else:
        b = np.logspace(np.log10(min(Y)), np.log10(max(Y)), nbins)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Delay Time [Myr]')
        ax.set_ylabel('PDF')
    entries, edges, _ = ax.hist(Y, bins=b, density=True, histtype='step', lw=3)

    # calculate bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])


    ax.set_title('Distribution of the Delay times')
    ax.grid(ls='dotted', lw=2)

    plt.show()


def Plot_TestPred(Y_pred, Y_test, log=False, save=False, filename=None):
    '''
    Produce the scatter plot of the predictions as a function
    of the true labels.

    Inputs:
        - Y_pred: array of the predictions
        - Y_test: array of the true labels
        - log: True if data are already logscaled, False if not
        - save: boolean: if true save figure
        - filename: name of the file to be saved
    '''

    fig, ax = plt.subplots(figsize=(15,12))
    ax.scatter(Y_test, Y_pred)
    ax.set_title('Y_pred vs Y_test')

    if log==True:
        ax.set_xlabel('Delay Time computed [Myr]')
        ax.set_ylabel('Delay Time predicted [Myr]')
        ax.grid(ls='dotted', lw=2)

        ax.set_xscale('log')
        ax.set_yscale('log')

    else:
        ax.set_xlabel('Test T_del [log/Myr]')
        ax.set_ylabel('Predicted T_del [log/Myr]')
        ax.grid(ls='dotted', lw=2)

    plt.show()
