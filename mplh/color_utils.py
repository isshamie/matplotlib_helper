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
               n_seq = 10) -> [dict, np.array]:
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
        return color_map, name_map

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




def create_color_df(meta_df, use_white: bool = False,
                    white_name: str = "N/A"):
    """Assumes each column is categorical for now. Creates a df with the same dimensions but now in its colors, along with the legend map."""
    meta_df_color = meta_df.copy()
    n_colors = 0
    labels = []
    for col in meta_df_color.columns.values:
        n_colors += len(meta_df[col].unique())
        labels += list(set(meta_df[col].values))
    color_map, name_map, p = get_colors("categorical", names=labels,
                                        n_colors=n_colors,
                                        use_white=use_white,
                                        white_name=white_name)
    for col in meta_df_color.columns.values:
        meta_df_color[col] = meta_df[col].map(color_map)

    return meta_df_color, color_map, name_map, p, labels
