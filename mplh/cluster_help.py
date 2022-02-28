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
                 row_clr_schemes='categorical', col_clr_schemes='categorical', **clust_kws):
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
        col_titles_d = {c: c for c in col_meta.columns}
        col_meta_color, col_labels_d, col_color_map = wrap_create_color_df_v02(col_meta, col_clr_schemes)
        # col_meta_color, col_color_map, name_map, p, labels = wrap_create_color_df(
        #     col_meta, use_white=True, white_name=white_name, scheme=scheme, sep_clr_map=False)
    else:
        col_meta_color = None

    if row_meta is not None:
        row_titles_d = {c:c for c in row_meta.columns}
        row_meta_color, row_labels_d, row_color_map = wrap_create_color_df_v02(row_meta, row_clr_schemes)
        # row_meta_color, row_color_map, name_map, p, labels = wrap_create_color_df(
        #     row_meta, use_white=True, white_name=white_name, scheme=scheme, sep_clr_map=False)
    else:
        row_meta_color = None

    #print('cmap', cmap)
    g = sns.clustermap(df, z_score=z, col_cluster=to_col_clust,
        row_cluster=to_row_clust, col_colors=col_meta_color,
        row_colors=row_meta_color, method=method,
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
        plot_legends(col_labels_d, col_color_map, col_titles_d,
                     ax=g.ax_col_dendrogram, loc="upper right")
        #legend_from_color(col_color_map, curr_ax=g.ax_col_dendrogram)
    if to_legend is not None and row_meta_color is not None:
        plot_legends(row_labels_d, row_color_map, row_titles_d,
                    ax=g.ax_row_dendrogram, loc='lower left')
        # legend_from_color(row_color_map,
        #                   curr_ax=g.ax_heatmap)  # g.ax_row_dendrogram)

    if name is not None:
        print('name', name)
        g.fig.suptitle(name)
    print("saving")
    if fsave is not None:
        fsave = fsave.replace(".png", "")
        g.savefig(fsave + ".png", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")
        g.savefig(fsave + ".svg", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")
        g.savefig(fsave + ".pdf", bbox_inches='tight', transparent=True,
                  markerfacecolor="None")
    return g


def extract_clusters(data, meta, g, axis, dist_thresh=0.6):
    import scipy
    from collections import defaultdict
    inds = g.dendrogram_row.dendrogram["leaves"]
    cols = g.dendrogram_col.dendrogram["leaves"]

    # data_clust = data.iloc[inds,cols]
    if axis == 0:
        meta = meta.iloc[inds]
    else:
        meta = meta.iloc[cols]

    den = scipy.cluster.hierarchy.dendrogram(g.dendrogram_row.linkage,
                                             labels=data.index,
                                             color_threshold=dist_thresh)

    def get_cluster_classes(den, label='ivl'):
        cluster_idxs = defaultdict(list)
        for c, pi in zip(den['color_list'], den['icoord']):
            for leg in pi[1:3]:
                i = (leg - 5.0) / 10.0
                if abs(i - int(i)) < 1e-5:
                    cluster_idxs[c].append(int(i))

        cluster_classes = {}
        for c, l in cluster_idxs.items():
            i_l = [den[label][i] for i in l]
            cluster_classes[c] = i_l

        return cluster_classes

    clusters = get_cluster_classes(den)

    cluster = []
    for i in data.index:
        included = False
        for j in clusters.keys():
            if i in clusters[j]:
                cluster.append(j)
                included = True
        if not included:
            cluster.append(None)

    meta.loc[data.index, "den_clust"] = cluster


def test_cluster():
    df = pd.DataFrame(np.random.randint(0,10,[10,20]))
    #col_
    return