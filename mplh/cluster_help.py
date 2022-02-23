import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import brewer2mpl
import sys
#sys.path.append("external/glasbey")

try:
    from mplh.fig_utils import legend_from_color
except ImportError:
    from .fig_utils import legend_from_color

print('here')

try:
    from mplh.color_utils import *
except ImportError:
    from .color_utils import *


try:
    # this works if you import Glasbey
    from mplh.glasbey import Glasbey
except ImportError:
    # this works if you run __main__() function
    from .glasbey import Glasbey


def plot_cluster(df: pd.DataFrame, row_meta=None, col_meta=None,
                 fsave=None, to_z=False, to_col_clust=True,
                 to_row_clust=True, name=None, col_names=True,
                 row_names=True, to_legend=True, method="average", white_name="WT", cmap=None, sep_clr_map=False,
                 scheme="categorical", **clust_kws):
    """Clusters dataframe, and includes different layers of metadata about the row and column da

    df: Dataframe to cluster on
    {row, col}_meta: meta DF with different groups. Indices are the df index IDs and the columns are different potential ways to group them
    name: Label of the graph
    fsave: Name to save the figure. Default is None, which will not save
    {row,col}_names: To keep the labels or not for the columns

    Returns: g, a seaborn ClusterGrid
    """

    z = None
    if to_z:
        z = 0

    if col_meta is not None:
        col_meta_color, col_color_map, name_map, p, labels = wrap_create_color_df(
            col_meta, use_white=True, white_name=white_name, scheme=scheme, sep_clr_map=False)
    else:
        col_meta_color = None

    if row_meta is not None:
        row_meta_color, row_color_map, name_map, p, labels = wrap_create_color_df(
            row_meta, use_white=True, white_name=white_name, scheme=scheme, sep_clr_map=False)
    else:
        row_meta_color = None

    print('cmap', cmap)
    g = sns.clustermap(df, z_score=z, col_cluster=to_col_clust,
        row_cluster=to_row_clust, col_colors=col_meta_color,
        row_colors=row_meta_color, method=method, cmap=cmap,
        **clust_kws)

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                 rotation=90, fontsize=8)

    if not col_names:
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_xticklabels("")
    if not row_names:
        g.ax_heatmap.set_yticks([])
        g.ax_heatmap.set_yticklabels("")

    if to_legend is not None and col_meta_color is not None:
        legend_from_color(col_color_map, curr_ax=g.ax_col_dendrogram)
    if to_legend is not None and row_meta_color is not None:
        legend_from_color(row_color_map,
                          curr_ax=g.ax_heatmap)  # g.ax_row_dendrogram)

    if name is not None:
        g.fig.suptitle(name)
    if fsave is not None:
        fsave = fsave.replace(".png", "")
        g.savefig(fsave + ".png", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")
        g.savefig(fsave + ".svg", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")
        g.savefig(fsave + ".pdf", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")

    return g


def test_cluster():
    df = pd.DataFrame(np.random.randint(0,10,[10,20]))
    #col_
    return