import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LoggerPlotter(object):
    def __init__(self, output_path, log_names, figure_names):
        self.dir_fig = os.path.join(output_path, 'figures')
        self.figures = {name: os.path.join(self.dir_fig, name)
                        for name in figure_names}

        self.dir_log = os.path.join(output_path, 'logs')
        os.makedirs(self.dir_fig)
        os.makedirs(self.dir_log)

        self.metrics = {}
        self.logs = {}
        for name in log_names:
            path = os.path.join(self.dir_log, name)
            self.logs.update({name: (path, open(path, "a"))})

    def log(self, log_name, line):
        self.logs[log_name][1].write(line + '\n')
        self.logs[log_name][1].flush()

    def add_line(self, line_name, log_name,
                 x_attribute, y_attribute,
                 color, linewidth=1, alpha=1):
        df = pd.read_csv(self.logs[log_name][0], sep=";")
        plt.figure(1, figsize=(10, 7), dpi=100, edgecolor='b')

        line, = plt.plot(df[x_attribute], df[y_attribute], label=line_name)
        plt.setp(line, color=color, linewidth=linewidth, alpha=alpha)

    def plot(self, fig_name, x_label, y_label, max_x=None, max_y=None):
        if max_x is not None:
            plt.xlim(xmax=max_x)
        if max_y is not None:
            plt.ylim(top=max_y)

        plt.axis([0, max_x, 0, max_y])
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.grid(alpha=0.6, linestyle='dotted')
        plt.legend()
        plt.savefig(self.figures[fig_name])
        plt.clf()

    def write_val_metrics(self, it, file):
        line = '{:.3f}'.format(it)
        for key, metric in self.metrics.items():
            line += (';' + metric.report_format).format(metric.val)
        self.log(file, line)

    def write_avg_metrics(self, it, file):
        line = '{:d}'.format(it)
        for key, metric in self.metrics.items():
            line += (';' + metric.report_format).format(metric.avg)
        self.log(file, line)

    def add_metric(self, metric_dict):
        for m in metric_dict:
            self.metrics.update({m.id_name: m})

    def print_metrics(self, epoch, iteration, total_iteration, phase='train'):
        phase_str = '@ Train ' if phase == 'train' else '# Test_ '
        sep = ' |' if phase == 'train' else ' /'
        epoch_str = '[{0:^5}-{1:>5}/{2:<5}]'.format(epoch, iteration, total_iteration)
        metric_str = ''
        for key, metric in self.metrics.items():
            tmp_str = (sep + ' {} = ' + metric.raw_format + ' (' + metric.avg_format + ')')
            metric_str += tmp_str.format(metric.display_name, metric.val, metric.avg)
        print(phase_str + epoch_str + metric_str)

    def clear_metrics(self):
        for key, metric in self.metrics.items():
            metric.reset()


class MetricLogger(object):
    """Compute and store the average and current value"""
    def __init__(self, id_name, display_name, raw_format=':.2e', avg_format=':.3e',
                 report_format=':.5f'):
        self.reset()
        self.id_name = id_name
        self.display_name = display_name
        self.raw_format = raw_format
        self.avg_format = avg_format
        self.report_format = report_format

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
