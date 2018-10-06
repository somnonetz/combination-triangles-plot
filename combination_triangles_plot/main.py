import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DESCRIPTION = 'Combination Triangles Plot'


def attach_args(parser):
    parser.add_argument(
        'ticks_file', action='store', type=str,
        help='CSV file with two columns and no header row. First column contains the axis ticks as used '
             'in the data index. Second column has tick translations to display in plot.'
    )
    parser.add_argument(
        'data_files', action='store', type=str, nargs='+',
        help='CSV files with three columns and a single header row. First column (index) contains the combined axis '
             'ticks. Second and third columns contain data to be plotted in lower left and upper right triangles '
             'respectively. Each CSV file is displayed in separate subplot. Data column names provided in header row '
             'are used in plot.'
    )
    parser.add_argument(
        '-t', '--titles-file', action='store', type=str, metavar='TITLES_FILE',
        help='CSV file with one column and no header row. Contains a subplot title for each data file in '
             'corresponding order.'
    )
    parser.add_argument(
        '-s', '--tick-separator', action='store', type=str, metavar='CHAR', default='-',
        help='Single CHAR, separating axis ticks in the data index.'
    )
    parser.add_argument(
        '-c', '--columns', action='store', type=int, metavar='COLUMNS', default=1,
        help='Number of COLUMNS for subplots.'
    )
    parser.add_argument(
        '--subplot-size', action='store', type=int, metavar='SUBPLOT_SIZE', default=10,
        help='Size of subplots.'
    )
    parser.add_argument(
        '--cbar-x', action='store', type=float, metavar='CBAR_X', default=.95,
        help='X position of color bar.'
    )
    parser.add_argument(
        '--cbar-y', action='store', type=float, metavar='CBAR_Y', default=.25,
        help='Y position of color bar.'
    )
    parser.add_argument(
        '--cbar-width', action='store', type=float, metavar='CBAR_WIDTH', default=.01,
        help='Width of color bar.'
    )
    parser.add_argument(
        '--cbar-height', action='store', type=float, metavar='CBAR_HEIGHT', default=.5,
        help='Height of color bar.'
    )
    parser.add_argument(
        '--label-ll-x', action='store', type=float, metavar='LABEL_LL_X', default=-3,
        help='X position of lower left (ll) label.'
    )
    parser.add_argument(
        '--label-ll-y', action='store', type=float, metavar='LABEL_LL_Y', default=39.5,
        help='Y position of lower left (ll) label.'
    )
    parser.add_argument(
        '--label-ur-x', action='store', type=float, metavar='LABEL_UR_X', default=38.7,
        help='X position of upper right (ur) label.'
    )
    parser.add_argument(
        '--label-ur-y', action='store', type=float, metavar='LABEL_UR_Y', default=-3.5,
        help='Y position of upper right (ur) label.'
    )
    parser.add_argument(
        '--vmin', action='store', type=float, metavar='VMIN',
        help='Scale minimum. Default is the minimum value across all plots.'
    )
    parser.add_argument(
        '--vmax', action='store', type=float, metavar='VMAX',
        help='Scale maximum. Default is the maximum value across all plots.'
    )


def main():
    parser = ArgumentParser(description=DESCRIPTION)
    attach_args(parser)
    args = parser.parse_args()

    return run(**args.__dict__)


def run(
        ticks_file,
        data_files,
        titles_file,
        tick_separator,
        columns,
        subplot_size,
        cbar_x,
        cbar_y,
        cbar_width,
        cbar_height,
        label_ll_x,
        label_ll_y,
        label_ur_x,
        label_ur_y,
        vmin,
        vmax
):
    if len(tick_separator) > 1:
        raise Exception('expected single character as TICK_SEPARATOR')

    if columns < 1:
        raise Exception('number of subplot COLUMNS must be at least 1')

    ticks = pd.read_csv(ticks_file, header=None, index_col=0)
    data = [pd.read_csv(f, index_col=0) for f in data_files]

    titles = [os.path.split(f)[1] for f in data_files]
    if titles_file:
        titles = pd.read_csv(titles_file, header=None)[0].values

    if len(titles) != len(data):
        raise Exception('number of titles must be equals number of DATA_FILES')

    vmin_total = float('inf')
    vmax_total = float('-inf')

    for df in data:
        df_plot, min_lower, max_lower, min_upper, max_upper, combined_mask = heatmap(
            df, ticks, tick_separator, combine_masks=True
        )
        vmin_total = min(min_lower, min_upper, vmin_total)
        vmax_total = max(max_lower, max_upper, vmax_total)

    if vmin is not None:
        vmin_total = vmin

    if vmax is not None:
        vmax_total = vmax

    center = 0
    cmap = 'RdBu'
    if vmin_total >= 0:
        center = None
        cmap = 'viridis'

    fontsize = 'large'

    num_subplots = len(data)
    rows = num_subplots // columns
    if num_subplots % columns > 0:
        rows += 1

    fig, axs = plt.subplots(ncols=columns, nrows=rows, figsize=(columns * subplot_size, rows * subplot_size))
    cbar = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    plt.subplots_adjust(hspace=0.25)

    for r in range(rows):
        for c in range(columns):
            idx = r * columns + c
            if idx >= len(data):
                break

            df = data[idx]
            title = titles[idx]

            if columns == 1 and rows == 1:
                ax = axs
            elif columns == 1:
                ax = axs[r]
            elif rows == 1:
                ax = axs[c]
            else:
                ax = axs[r][c]

            ax.set_title(title)
            ax.title.set_position([.5, 1.09])

            label_ll, label_ur = df.columns[:2]

            df_plot, min_lower, max_lower, min_upper, max_upper, combined_mask = heatmap(
                df, ticks, tick_separator, combine_masks=True
            )

            cbar_ax = None
            enable_cbar = False

            if r == 0 and c == columns - 1:
                cbar_ax = cbar
                enable_cbar = True

            sns.heatmap(df_plot, mask=combined_mask, ax=ax, square=True, cmap=cmap, center=center, vmin=vmin_total,
                        vmax=vmax_total, cbar=enable_cbar, cbar_ax=cbar_ax)
            ax.tick_params(labeltop=True, labelright=True, labelrotation='default')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.text(label_ll_x, label_ll_y, label_ll, fontsize=fontsize, rotation=45)
            ax.text(label_ur_x, label_ur_y, label_ur, fontsize=fontsize, rotation=45)

    plt.savefig('plot.pdf', bbox_inches='tight')


def heatmap(df, ticks, tick_separator, combine_masks=False):
    ticks_map = {c: i for i, c in enumerate(ticks.index)}

    df_plot = pd.DataFrame(0.0, index=ticks.index, columns=ticks.index)
    df_mask_lower = pd.DataFrame(1, index=ticks.index, columns=ticks.index)
    df_mask_upper = pd.DataFrame(1, index=ticks.index, columns=ticks.index)

    mins_maxs = []

    for i, column in enumerate(df.columns[:2]):
        column_min = float('inf')
        column_max = float('-inf')

        for row_index in df.index:
            t0, t1 = row_index.split(tick_separator)
            t0_i = ticks_map[t0]
            t1_i = ticks_map[t1]

            if t0_i > t1_i:
                r, c = t0, t1
            else:
                r, c = t1, t0

            mask = df_mask_lower
            if i == 1:
                mask = df_mask_upper
                r, c = c, r

            val = df[column][row_index]
            df_plot[c][r] = val
            mask[c][r] = 0

            column_min = min(column_min, val)
            column_max = max(column_max, val)

        mins_maxs.append((column_min, column_max))

    df_plot.index = ticks[1]
    df_plot.columns = ticks[1]

    mask_lower = df_mask_lower.values
    mask_upper = df_mask_upper.values

    (min_lower, max_lower), (min_upper, max_upper) = mins_maxs

    if combine_masks:
        combined_mask = mask_lower + mask_upper - np.ones((ticks.shape[0], ticks.shape[0]))
        return df_plot, min_lower, max_lower, min_upper, max_upper, combined_mask

    return df_plot, min_lower, max_lower, min_upper, max_upper, mask_lower, mask_upper
