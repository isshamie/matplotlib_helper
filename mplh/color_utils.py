import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from .fig_utils import legend_from_color
import brewer2mpl
#from .glasbey import Glasbey
from matplotlib.patches import Patch


try:
    # this works if you import Glasbey
    from mplh.glasbey import Glasbey
except ImportError:
    # this works if you run __main__() function
    from .glasbey import Glasbey


def get_colors(scheme: str, names=None, n_colors:int = -1, use_white:bool=False,
               white_name:str = "N/A", use_black:bool=False,
               black_name:str = "N/A",
               n_seq = 10,return_p=False) -> [dict, np.array]:
    """
    :param scheme: {'sequential', 'divergent', 'categorical'}
    :param n_colors: int
        Number of colors needed
    :param use_white: bool
        If True, white will also be geneated, which will be the first index.
    :param use_black: bool
        If True, white will also be geneated, which will be the second index.
    :return:
    """
    if scheme == "categorical":
        if names is None:
            print("Need a list of names to create dictionary")
            raise TypeError
        gb = Glasbey()
        p = gb.generate_palette(size=n_colors + 2)
        # Needed for Float problems
        p[p > 1] = 1
        p[p < 0] = 0
        color_map = {name: p[ind + 2] for ind, name in
                     enumerate(set(names))}
        if use_white:
            color_map[white_name] = p[0]
        if use_black:
            color_map[black_name] = p[1]

        name_map = {val: ind for ind, val in enumerate(names)}
        if return_p:
            return color_map, name_map,p
        else: return color_map, name_map
    elif scheme == 'divergent':
        bmap = brewer2mpl.get_map('Divergent', 'Blues', n_seq)
        color_map = bmap.mpl_colormap

    elif scheme == "sequential":
        bmap = brewer2mpl.get_map('sequential', 'Qualitative', 5)
        color_map = bmap.mpl_colormap
    else:
        print("scheme needs to be from catgorical, divergent, or sequential")
        raise TypeError
    return color_map



def wrap_create_color_df(meta_df, use_white: bool = False,
                         white_name: str = "N/A", scheme="categorical", sep_clr_map=False):
    if sep_clr_map:
        return create_color_df(meta_df, use_white, white_name, scheme=scheme)
    else:
        return create_color_df_sepMap(meta_df, use_white, white_name, scheme=scheme)


def create_color_df(meta_df, use_white: bool = False,
                    white_name: str = "N/A", scheme="categorical"):
    """Assumes each column is categorical for now. Creates a df with the same dimensions but now in its colors, along with the legend map."""
    meta_df_color = meta_df.copy()
    n_colors = 0
    labels = []
    for col in meta_df_color.columns.values:
        n_colors += len(meta_df[col].unique())
        labels += list(set(meta_df[col].values))
    color_map, name_map, p = get_colors(scheme, names=labels,
                                        n_colors=n_colors,
                                        use_white=use_white,
                                        white_name=white_name, return_p=True)

    for col in meta_df_color.columns.values:
        meta_df_color[col] = meta_df[col].map(color_map)

    return meta_df_color, color_map, name_map, p, labels


def create_color_df_sepMap(meta_df, use_white: bool = False,
                    white_name: str = "N/A", scheme=("categorical")):
    """TODO: Allows for separate color maps for each column and separate scehmes"""
    meta_df_color = meta_df.copy()
    all_labels = []
    all_p = []
    all_color_map = {}
    all_cols = meta_df_color.columns.values
    if type(scheme) == str:
        scheme = {i:scheme for i in all_cols}

    for col in meta_df_color.columns.values:
        n_colors = len(meta_df[col].unique())
        labels = list(set(meta_df[col].values))
        color_map, name_map, p = get_colors(scheme[col], names=labels,
                                            n_colors=n_colors,
                                            use_white=use_white,
                                            white_name=white_name,return_p=True)
        for c in color_map:
            if c in all_color_map:
                all_color_map[f"{col}_{c}"] = color_map[c]
            else:
                all_color_map[c] = color_map[c]
        meta_df_color[col] = meta_df[col].map(color_map)
        all_labels.append(labels)
        all_p.append(p)
    return meta_df_color, all_color_map, name_map, p, labels


def wrap_create_color_df_v02(meta_df, clr_types_d):
    clr_map_df = meta_df.copy() # Colors
    anno_labels_d ={}
    anno_lut_d = {}

    # Generate color map keys:
    #print(clr_types_d, type(clr_types_d) == str)
    if type(clr_types_d) == str:
        clr_types_d = {x:clr_types_d for x in meta_df.columns}

    clr_keys = {}
    seq_cnt, div_cnt, cat_cnt = 0, 0, 0
    for c in clr_types_d:
        if (clr_types_d[c])=='sequential':
            seq_cnt+=1
            clr_keys[c] = seq_cnt
        elif (clr_types_d[c])=='divergent':
            div_cnt+=1
            clr_keys[c] = div_cnt
        elif (clr_types_d[c])=='categorical':
            cat_cnt+=1
            clr_keys[c] = cat_cnt

    for c in meta_df.columns:
        clr_map, anno_labels, anno_lut = create_color_df_v02(meta_df, c,
                                                             clr_type=clr_types_d[c],
                                                             clr_key=clr_keys[c],
                                                             add_suffix=False)
        clr_map_df[c] = clr_map
        anno_labels_d[c] = anno_labels
        anno_lut_d[c] = anno_lut
    return clr_map_df, anno_labels_d, anno_lut_d


def get_glasbey_colors(n_colors, names, white_name:str = "N/A", use_black:bool=False,
                       black_name:str = "N/A", use_white:bool=False, palette=sns.color_palette('Dark2')):
    if names is None:
        print("Need a list of names to create dictionary")
        raise TypeError
    names = [str(x) for x in names]
    gb = Glasbey(base_palette=palette)
    anno_pal = gb.generate_palette(size=n_colors + 2)
    # Needed for Float problems
    anno_pal[anno_pal > 1] = 1
    anno_pal[anno_pal < 0] = 0
    color_map = {name: anno_pal[ind + 2] for ind, name in
                 enumerate(set(names))}
    if use_white:
        color_map[white_name] = anno_pal[0]
    if use_black:
        color_map[black_name] = anno_pal[1]
    return color_map


def create_color_df_v02(meta_df, col, clr_key=1, clr_type='sequential',
                        add_suffix=True):
    """ Create color values for each column in meta_df

    :param meta_df: DataFrame where each index is an observation and each column is a feature
    :param col: Which column to use in dataframe
    :param clr_key: int to map to the color maps defined here
    :param clr_type: {'sequential', 'divergenet', 'categorical'
    :return:
    """

    # First get the labels
    anno_labels = np.sort(meta_df[col].unique())

    # Create the palettes.
    seq_clr_keys = {
        1: sns.cubehelix_palette(len(anno_labels), light=.9, dark=.2,
                                 reverse=True, rot=.1, start=2.8),
        2: sns.cubehelix_palette(len(anno_labels), light=.9, dark=.2,
                                 reverse=True, rot=.1, start=4.2),
        3: sns.cubehelix_palette(len(anno_labels), light=.9, dark=.2,
                                 reverse=True, rot=.3, start=0)}
    divergent_clr_keys = {
        1: sns.diverging_palette(240, 10, n=len(anno_labels)),
        2: sns.diverging_palette(150, 275, s=80, l=55,
                                 n=len(anno_labels))}
    cat_clr_keys = {1: sns.color_palette("Set2"),
                    2: sns.color_palette("Dark2"),
                    3: sns.color_palette("Paired"),
                    4: sns.color_palette("tab10")}
    # cat_clr_keys = {1: "dark2",
    #                 2: "set1",
    #                 3: "set2"}

    if clr_type == "sequential":
        anno_pal = seq_clr_keys[((clr_key-1) % len(seq_clr_keys)+1)]
        anno_lut = dict(zip(map(str, anno_labels), anno_pal))
    elif clr_type == "divergent":
        anno_pal = divergent_clr_keys[((clr_key-1)%len(divergent_clr_keys)+1)]
        anno_lut = dict(zip(map(str, anno_labels), anno_pal))
    elif clr_type == "categorical":
        base_pal = cat_clr_keys[((clr_key-1)%len(cat_clr_keys)+1)]
        anno_lut = get_glasbey_colors(len(anno_labels), anno_labels, palette=base_pal)
    else:
        raise ValueError

    # anno_lut relates the labels to the colors, and anno_colors just puts into series
    anno_colors = pd.Series(anno_lut)
    # Create series of index to color
    clr_ser = meta_df[col].apply(lambda x: anno_colors.loc[str(x)])

    if add_suffix:
        meta_df[f"{col}_map"] = clr_ser
        return meta_df, anno_labels, anno_lut
    else:
        return clr_ser, anno_labels, anno_lut


def plot_legends(labels_d, lut_d, titles_d, ax=None, scheme=None, axis="row"):
    if ax is None:
        ax = plt.gca()

    legends = []
    if axis=="row":
        box_keys = [1-(i*1/len(labels_d)) for i in range(len(labels_d))]
    else:
        box_keys = [1-(i*1/len(labels_d)) for i in range(len(labels_d))]

    count = 0
    if type(scheme) == str:
        scheme = {i:scheme for i in labels_d.keys()}
    #print('dat_type', scheme)
    for d in labels_d:
        # create_legend(labels_d[d], lut_d[d],
        #               n_labs=8, title=titles_d[d], ax=ax, loc=loc)

        if scheme is not None and scheme[d] == "categorical":
            #print(scheme[d])
            if len(labels_d[d]) > 16:
                print("Using only 16 of the labels for legend")
            n_labs = 16 # len(labels_d[d])
        else:
            n_labs = 8
        if axis == "row":
            legends.append(create_legend(labels_d[d], lut_d[d],
                            n_labs=n_labs, title=titles_d[d], ax=ax, bbox_y=box_keys[count], bbox_x=1.4))
        else:
            legends.append(create_legend(labels_d[d], lut_d[d], n_labs=n_labs, title=titles_d[d], ax=ax, bbox_x=box_keys[count], bbox_y=1.4))
        count+=1
        ax.add_artist(legends[-1]) #, loc=loc, bbox_to_anchor=(1.1, 0.5))
    return


def create_legend(anno_labels, anno_lut, title, bbox_x=1.4, bbox_y=0.5, n_labs=-1, ax=None):
    if ax is None:
        ax = plt.gca()
    handles = []
    if n_labs == -1 or n_labs > len(anno_labels):
        n_labs = len(anno_labels)

    step = int(np.round(len(anno_labels) / n_labs))
    labels = []
    for label in anno_labels[::step]:
        if type(label) == str:
            nm = f'{label}'
        else:
            nm=f'{label:.3g}'

        handles.append(Patch(facecolor=[float(x) for x in anno_lut[str(label)]], label=nm))
        labels.append(nm)

    curr_legend = ax.legend(handles=handles, loc="center", title=title, ncol=2,
                            bbox_to_anchor=(bbox_x, bbox_y),)
    return curr_legend


def wrap_legends():
    return




def plot_continuous_legend(g, anno_labels, anno_lut, n_labs=-1,
                           title=None, loc='right'):
    if n_labs == -1 or n_labs > len(anno_labels):
        n_labs = len(anno_labels)

    step = int(np.round(len(anno_labels) / n_labs))

    # if title is not None:
    #     tit = g.ax_heatmap.bar(0, 0,label=title, linewidth=0)
    for label in anno_labels[::step]:
        if type(label) == str:
            g.ax_heatmap.bar(0, 0, color=anno_lut[str(label)],
                             label=f'{label}', linewidth=0)
        else:
            g.ax_heatmap.bar(0, 0, color=anno_lut[str(label)],
                             label=f'{label:.3g}',
                             linewidth=0)  # g.ax_col_dendrogram.bar(0, 0, color=anno_lut[str(label)],  #                         label=label, linewidth=0)  # plt.bar(0, 0, color=anno_lut[str(label)],  #                 label=label, linewidth=0)

    g.ax_heatmap.legend(bbox_to_anchor=(1.4, 1.2), ncol=4, loc=loc,
                        borderaxespad=1)
    # g.ax_col_dendrogram.legend(ncol=4 )#loc="best", ncol=6)
    return g.ax_heatmap.legend()
