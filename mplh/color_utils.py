import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from .fig_utils import legend_from_color
import brewer2mpl
#from .glasbey import Glasbey

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