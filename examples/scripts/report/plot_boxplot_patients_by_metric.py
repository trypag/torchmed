import argparse
import os
import numpy as np
import pandas
import matplotlib as mpl

# agg backend is used to create plot as a .png file
mpl.use('agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Evaluation of metrics with boxplots')
parser.add_argument('data', metavar='DIR',
                    help='path to root segmentation dir')
parser.add_argument('output', metavar='DIR',
                    help='path to output dir')


def main():
    global args
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # iterate over segmentation folders to gather data
    for file in os.listdir(args.data):
        filepath = os.path.join(args.data, file)
        if os.path.isdir(filepath) and not file.startswith('__'):
            metric_file = os.path.join(filepath, 'metrics_report.csv')
            col_names = pandas.read_csv(metric_file, sep=';', index_col=0).columns.values
            break

    for metric_name in col_names:
        df = None
        p_names = []
        # iterate over segmentation folders to gather data
        for file in os.listdir(args.data):
            filepath = os.path.join(args.data, file)
            if os.path.isdir(filepath) and not file.startswith('__'):
                p_names.append(file)
                metric_file = os.path.join(filepath, 'metrics_report.csv')
                patient_m = pandas.read_csv(metric_file, sep=';', index_col=0)[metric_name]
                if df is None:
                    df = patient_m
                else:
                    df = pandas.concat([df, patient_m], axis=1)
        df.columns = p_names

        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                figsize=(50, 10), dpi=100)
        plt.subplots_adjust(hspace=0.2)

        plot_boxplot(ax1, 111, df,
                     'boxplot of the ' + metric_name + ' for each patients',
                     'patients',
                     metric_name,
                     None, None)

        outfile = os.path.join(args.output, 'boxplot_patients_by_' + metric_name + '.png')
        fig.savefig(outfile, bbox_inches='tight')
        plt.clf()


def read_and_aggregate(filename, df=None):
    '''
    Read a csv file and aggregates the result into a pandas dataframe
    '''

    read_df = pandas.read_csv(filename, sep=';', index_col=0)

    if df is None:
        return read_df
    else:
        return pandas.concat([df, read_df], axis=1)


def plot_boxplot(ax, d, df, title, x_axis, y_axis,
                 y_scale=None, y_precision=None):
    samples = []
    # for index, row in df.iterrows():
    #     samples.append(row.tolist())
    for col in df:
        samples.append(df[col].tolist())

    # Create the boxplot
    bp = ax.boxplot(samples, patch_artist=True)

    # change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')

    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # set y axis scale
    if y_scale is not None:
        assert(isinstance(y_scale, list))
        start, end = y_scale
    else:
        start, end = ax.get_ylim()
        start = 0

    ax.set_ylim((start, end + ((end - start) / 20)))

    # increase y axis precision
    if y_precision is not None:
        ax.yaxis.set_ticks(np.arange(start, end, y_precision))

    # add grid
    ax.grid(True, color='lightgrey', alpha=0.5)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()

    # rotate labels on the x axis
    ax.set_xticklabels(df.columns, rotation=35)

    # set title
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    # add mean value on top on each label column
    pos = np.arange(len(samples)) + 1
    means = [np.mean(s) for s in samples]
    upperLabels = [str(np.round(s, 2)) for s in means]
    upperLabels = [l[:4] for l in upperLabels]
    weights = ['bold', 'semibold']
    boxColors = ['darkkhaki', 'royalblue']
    top = end + ((end - start) / 20)
    for tick, label in zip(range(len(samples)), ax.get_xticklabels()):
        k = tick % 2
        ax.text(pos[tick], top - (top * 0.05), upperLabels[tick],
                horizontalalignment='center', size='x-small', weight=weights[k],
                color='#7570b3')


if __name__ == '__main__':
    main()
