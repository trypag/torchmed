import argparse
import os
import numpy as np
import pandas

parser = argparse.ArgumentParser(
    description='Make model reports')
parser.add_argument('data', metavar='DIR',
                    help='path to the segmentation dir')
parser.add_argument('output', metavar='OUT',
                    help='path to the output dir')
parser.add_argument('metric', metavar='METRIC NAME',
                    help='name of the metric')
parser.add_argument('-k', '--k-top', metavar='N', default=5, type=int,
                    help='number of element to retain')
parser.add_argument('-t', '--is-top', action='store_false',
                    help='whether to use k-top or not (k-small)')


def main():
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    df = None
    undetected_classes = []
    for seg in os.listdir(args.data):
        seg_dir = os.path.join(args.data, seg)
        if os.path.isdir(seg_dir):
            report_file = os.path.join(seg_dir, 'metrics_report.csv')
            df = read_and_aggregate(report_file, df)

    # name of each metric
    metrics_name = sorted(list(set(df.columns.values)))

    if args.metric in metrics_name:
        m_df = df[args.metric].mean(1)
        std_df = df[args.metric].std(1)

        idx = np.argsort(m_df)
        idx = idx[:args.k_top] if args.is_top else list(reversed(idx))[:args.k_top]

        # write mean and std of classes to csv
        f_name = str(args.k_top) + ('_smallest_' if args.is_top else '_highest_') + args.metric
        file_name = os.path.join(args.output, f_name + '.csv')
        mean_std = []
        for i in idx:
            mean_std.append('{:.3f} += {:.2f}'.format(m_df[i], std_df[i]))
        o_df = pandas.DataFrame({'mean': mean_std, 'label_id': idx})
        o_df.to_csv(
            file_name, sep=';',
            index=True, index_label='order')


def read_and_aggregate(filename, df=None):
    read_df = pandas.read_csv(filename, sep=';', index_col=0)

    if df is None:
        return read_df
    else:
        return pandas.concat([df, read_df], axis=1)


if __name__ == '__main__':
    main()
