import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import pickle
import pandas as pd
import glob
import os
from os.path import join
import sys
from .glasbey import Glasbey
from matplotlib import rcParams
from os.path import basename
import matplotlib as mpl
import brewer2mpl

#rcParams['figure.figsize'] = 8, 6
#mpl.style.use('ggplot')
#mpl.style.use('fivethirtyeight')
#from cycler import cycler
sys.setrecursionlimit(3000)


def helper_save(f_save=None, remove_bg=True, to_svg=True, dpi=300):
    """
    Function to save figure file as png and svg if f_save is not None. Creates a
    transparent background.
    :param f_save : str
        Filename to save in. Will replace .csv, .txt, .tsv if appended. If f_save is None, will not save.
    :param remove_bg : bool
        If True, remove the background of the plot
    :param to_svg : bool
        If true, save as an svg
    :return:
    """
    if remove_bg:
        f = plt.gcf()
        f.patch.set_facecolor("w")
        axs = f.get_axes()
        for ax in axs:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    if f_save is not None and not f_save == "":
        #Remove suffixes
        f_save = f_save.replace('.csv', '')
        f_save = f_save.replace('.tsv', '')
        f_save = f_save.replace('.txt', '')
        f_save = f_save.replace(".png", "")

        # Saving
        plt.savefig(f_save+".png", bbox_inches="tight", transparent=True,
                    dpi=dpi, pad_inches=0.1)

        if to_svg:
            plt.savefig(f_save + '.svg')
    else:
        print("No file to save in.")
    return


def legend_from_color(color_map, curr_ax, f=None):
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o',
                          linestyle='') for color in color_map.values()]
    curr_ax.legend(markers, list(color_map.keys()),
                   bbox_to_anchor=(1.15, 0.75)) # numpoints=1, loc='upper right')
    # curr_ax.legend(markers, list(color_map.keys()),bbox_to_anchor = (1.15, 1),
    #                 bbox_transform = plt.gcf().transFigure)
    return


def plot_marginals():
    ### TO DO  ###
    ### Make a gridspec and be able to alter the dimensions
    return


def num_rows_cols(n, max_cols=5, nrows=None, ncols=None):
    """
    Function to determine the number of rows and columns.
    :param n: int
        Number of subplots to create
    :param max_cols: int
        Maximum number of columns to have. Default is 5
    :param nrows, ncols: If set, will hard-set the number of rows and fill in the number of rows {columns}.
                         If only one is set will determine the rest based on n.
    :return:
    """

    # If nrows is s
    if nrows is not None:
        if ncols is not None:
            return nrows, ncols
        else:
            return nrows, ceil(n/nrows)
    elif ncols is not None:
        return ceil(n/ncols), ncols
    # Would rather more rows than cols, but would rather be even
    elif ceil(n/ceil(np.sqrt(n))) >= max_cols:
        return ceil(n/max_cols), max_cols
    else:
        return ceil(np.sqrt(n)), ceil(n/ceil(np.sqrt(n)))


def heatmap_control_color_bar(df):
    """ TO FINISH
        Taken 12/1"""
    # Define two rows for subplots
    fig, (cax, ax) = plt.subplots(nrows=2, figsize=(5, 5.025),
                                  gridspec_kw={
                                      "height_ratios": [0.025, 1]})

    # Draw heatmap
    sns.heatmap(df, ax=ax, cbar=False)

    # colorbar
    fig.colorbar(ax.get_children()[0], cax=cax,
                 orientation="horizontal")
    return


def stack_bars(x, ys, ylabels, f_save=None, use_text=True,deci=0):
    # stack bars
    f = plt.figure()
    for ind, curr_y in enumerate(ys):
        plt.bar(x,curr_y , bottom= ys[:ind].sum(axis=0), label=ylabels[ind])

    # add text annotation corresponding to the percentage of each data.
    if use_text:
        for xpos, ypos, yval in zip(x, ys[0]/2, ys[0]):
            plt.text(xpos, ypos, f"%.{deci}f"%(yval*100), ha="center", va="center")

        for ind, val in enumerate(ys):
            if ind == 0:
                continue
            for xpos, ypos, yval in zip(x,ys[:ind].sum(axis=0)+val/2, val):
                plt.text(xpos, ypos, f"{(yval*100):0.{deci}f}", ha="center", va="center")

    #plt.ylim(0,110)
    plt.title("Normalized stacked barplot of different RNA types")
    plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left')
    if f_save is not None:
        helper_save(f_save)
    plt.xticks(rotation=90)
    #plt.savefig('normalized_stacked_barplot_with_number.png', bbox_inches='tight', pad_inches=0.02)
    return
print("fig_utils from mplh")


def df_stacked_bar(files, column="annotation", f_save=None,use_text=True):
    """ Creates a stacked barplot for each file, based on the column of choice."""
    anno_labels = set()
    anno_count = dict()
    for f in files:
        print(f)
        df = pd.read_csv(f, sep="\t", index_col=0)
        anno_labels = anno_labels.union(set(df[column].unique()))
        if len(df) == 0:
            print("Empty")
        else:
            anno_count[os.path.basename(f)] = df.groupby(
                column).count().iloc[:, 0].to_dict()
    x = np.array(list(anno_labels))
    ys = []
    ys = pd.DataFrame(index=x, columns=anno_count.keys(), dtype=int)
    ys.loc[:, :] = 0
    for s in anno_count:
        for type_count in anno_count[s]:
            ys.at[type_count, s] = anno_count[s][type_count]
    stack_bars(ys.columns.values, ys=(ys / (ys.sum(axis=0))).to_numpy(),
               ylabels=x,use_text=use_text)
    plt.title(column)

    helper_save(f_save)
    ys.to_csv(f_save+'.csv')
    return anno_count


def show_values_on_bars(axs,deci=2):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = f'{p.get_height():0.{deci}f}'
            #value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)