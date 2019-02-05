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

    models = os.listdir(args.data)
    models_df = []
    # for each model
    for model in models:
        model_dir = os.path.join(args.data, model)
        if os.path.isdir(model_dir):

            # collect class metrics
            df = None
            undetected_classes = []
            data_dir = os.path.join(model_dir, 'segmentations')
            for seg in os.listdir(data_dir):
                seg_dir = os.path.join(data_dir, seg)
                if os.path.isdir(seg_dir):
                    report_file = os.path.join(seg_dir, 'metrics_report.csv')
                    undec_classes = os.path.join(seg_dir, 'undetected_classes.csv')
                    df = read_and_aggregate(report_file, df)
                    if os.path.isfile(undec_classes):
                        undetected_classes.append((seg, read_and_aggregate(undec_classes)['class_id'].tolist()))
            models_df.append((model, df, undetected_classes))

    # name of each metric
    metrics_name = sorted(list(set(df.columns.values)))
    log = open(os.path.join(args.output, 'log.csv'), "w")
    models_df = sorted(models_df, key=lambda k:k[0])

    # average of metrics for each model
    log.write('model;{}'.format(';'.join(metrics_name)))
    for name, df, undetected in models_df:
        # mean by metric over columns
        log.write('\n{}'.format(name))
        for metric in metrics_name:
            m_df = df[metric].mean(1)

            # write mean and std of each metric to file
            avg = m_df.mean(0)
            std = m_df.std(0)
            log.write(';{:.3f} += {:.3f}'.format(avg, std))
    log.write('\n')

    for name, df, undetected in models_df:
        if len(undetected_classes) > 0:
            log.write('\n {} Undetected Structures:\n'.format(name))
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
