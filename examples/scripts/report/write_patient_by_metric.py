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

    avgs = None
    patient_name = []
    patients = sorted(os.listdir(args.data))
    for seg in patients:
        seg_dir = os.path.join(args.data, seg)
        if os.path.isdir(seg_dir):
            patient_name.append(seg)
            report_file = os.path.join(seg_dir, 'metrics_report.csv')
            m_df = pandas.read_csv(report_file, sep=';', index_col=0)

            if avgs is None:
                avgs = m_df.mean(0).values
                stds = m_df.std(0).values
            else:
                avgs = np.vstack((avgs, m_df.mean(0).values))
                stds = np.vstack((stds, m_df.std(0).values))

    nb_patient, nb_metric = avgs.shape
    metrics = [[0] * nb_metric] * nb_patient
    for p in range(nb_patient):
        for m in range(nb_metric):
            p_list = metrics[p]
            p_list[m] = '{:.3f} += {:.2f}'.format(avgs[p, m], stds[p, m])
            metrics[p] = p_list.copy()

    df = pandas.DataFrame(metrics)
    df.columns = [m_df.columns.values]
    df['name'] = patient_name

    # reorder columns to have name in first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # name of each metric
    report_file = os.path.join(args.output, 'metric_by_patient.csv')
    df.to_csv(report_file, sep=";", index=False)


if __name__ == '__main__':
    main()
