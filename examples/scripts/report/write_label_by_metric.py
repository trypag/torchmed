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
            undec_classes = os.path.join(seg_dir, 'undetected_classes.csv')
            df = read_and_aggregate(report_file, df)
            if os.path.isfile(undec_classes):
                undetected_classes.append((seg, read_and_aggregate(undec_classes)['class_id'].tolist()))

    # name of each metric
    metrics_name = sorted(list(set(df.columns.values)))
    log = open(os.path.join(args.output, 'average_metrics.txt'), "w")

    # mean by metric over columns
    for metric in metrics_name:
        m_df = df[metric].mean(1)
        std_df = df[metric].std(1)
        # write mean and std of classes to csv
        file_name = os.path.join(args.output, metric + '_by_label.csv')
        mean_std = []
        for n in range(m_df.shape[0]):
            mean_std.append('{:.3f} += {:.2f}'.format(m_df[n], std_df[n]))
        pandas.DataFrame(mean_std).to_csv(
            file_name, sep=';', header=['mean += std'],
            index=True, index_label='label_id')

        # write mean and std of each metric to file
        avg = m_df.mean(0)
        std = m_df.std(0)
        log.write('{}: {:.3f} += {:.3f}\n'.format(metric, avg, std))

    if len(undetected_classes) > 0:
        log.write('\nUndetected Structures:\n')
        for seg, classes in undetected_classes:
            log.write('{}: {}\n'.format(seg, ', '.join(map(str, classes))))
    log.flush()


def read_and_aggregate(filename, df=None):
    read_df = pandas.read_csv(filename, sep=';', index_col=0)

    if df is None:
        return read_df
    else:
        return pandas.concat([df, read_df], axis=1)


if __name__ == '__main__':
    main()
