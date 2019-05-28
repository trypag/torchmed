import os
import time
import torch
import pandas as pd
import collections

from torchmed.utils.metric import multiclass, dice, hd, assd, precision
from torchmed.readers import SitkReader


class InferenceCanvas(object):
    def __init__(self, args, inference_fn, data_fn, model):
        self.args = args
        self.dataset_fn = data_fn
        self.inference_fn = inference_fn
        self.model = model

        self.file_names = {'image': 'prepro_im_mni_bc.nii.gz',
                           'label': 'prepro_seg_mni.nii.gz',
                           'segmentation': 'segmentation.nii.gz'}
        self.metrics = {'dice': dice,
                        'hausdorff': hd,
                        'mean_surface_distance': assd,
                        'precision': precision
                        }

    def __call__(self):
        print("=> started segmentation script")
        # if output dir does not exists, create it
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output)

        print("=> loading the architecture")
        model = self.model.cuda()

        print("=> loading trained model at {}".format(self.args.model))
        # load model parameters
        torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(self.args.model)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            model = model.module

        print("=> segmentation output at {}".format(self.args.output))
        test_times = []
        if os.path.isfile(os.path.join(self.args.data, 'allowed_data.txt')):
            allowed_data_file = open(
                os.path.join(self.args.data, 'allowed_data.txt'), 'r')
            patient_list = [line.rstrip('\n') for line in allowed_data_file]

            for patient in patient_list:
                if patient:
                    patient_dir = os.path.join(self.args.data, patient)
                    patient_out = os.path.join(self.args.output, patient)
                    os.makedirs(patient_out)
                    test_time = self.segment_metric_plot(model,
                                                         patient_dir,
                                                         patient_out)
                    test_times.append(test_time)

        else:
            test_times.append(
                self.segment_metric_plot(model, self.args.data,
                                         self.args.output))

        # write test time to file
        time_file = os.path.join(self.args.output, 'test_time.csv')
        time_report = open(time_file, 'a')
        time_report.write('image_id;minutes\n')
        for time_id in range(0, len(test_times)):
            time_report.write('{};{:.5f}\n'.format(
                time_id, test_times[time_id]))
        time_report.write('{};{:.5f}\n'.format(
            'average', sum(test_times) / len(test_times)))
        time_report.flush()

    def segment_metric_plot(self, model, patient_dir, patient_out):
        patient_seg = os.path.join(patient_out,
                                   self.file_names['segmentation'])

        # segmentation
        test_time = self.segment_one_patient(model, patient_dir, patient_out)
        print('-- segmented {} in {:.2f}s'.format(patient_dir, test_time))

        # if ground truth is available, use metrics
        if self.args.wo_metrics:
            patient_map = os.path.join(patient_dir,
                                       self.file_names['label'])

            self.save_error_map(patient_dir, patient_out)

            # evaluate metrics
            ref_img = SitkReader(patient_map).to_numpy()
            seg_img = SitkReader(patient_seg).to_numpy()
            results, undec_structs = multiclass(seg_img, ref_img,
                                                self.metrics.values())
            metrics_results = zip(self.metrics.keys(), results)

            m = collections.OrderedDict(sorted(metrics_results, key=lambda x: x[0]))
            df = pd.DataFrame.from_dict(m)
            df.to_csv(os.path.join(patient_out, 'metrics_report.csv'), ';')

            if len(undec_structs) > 0:
                df = pd.DataFrame(undec_structs, columns=["class_id"])
                df.to_csv(os.path.join(patient_out, 'undetected_classes.csv'), ';')

        return test_time

    def segment_one_patient(self, model, data, output):
        # Data loading code
        medcomp = self.dataset_fn(data, self.args.batch_size).test_data
        loader = torch.utils.data.DataLoader(medcomp,
                                             batch_size=self.args.batch_size,
                                             shuffle=False,
                                             num_workers=5,
                                             pin_memory=True)

        lab = SitkReader(os.path.join(data, self.file_names['image']),
                         torch_type='torch.LongTensor')
        lab_array = lab.to_numpy()
        lab_array.fill(0)

        start_time = time.time()
        probability_maps = self.inference_fn(model, loader, lab_array)
        end_time = time.time()

        # save label map
        lab.to_image_file(os.path.join(output, self.file_names['segmentation']))

        """
         generation of probability maps, use only if you want to visualize
         probabilities for a given label, otherwise it will generate all
         the probabilities maps
        """
        if len(probability_maps) > 0:
            os.makedirs(os.path.join(output, "probability_maps"))

            img = SitkReader(os.path.join(data, self.file_names['image']))
            img_array = img.to_numpy()
            for map_id in range(0, len(probability_maps)):
                prob_file = os.path.join(output,
                                         "probability_maps/label_{}.img".format(map_id))
                img_array.fill(0)
                img_array[...] = probability_maps[map_id]
                img.to_image_file(prob_file)

        return (end_time - start_time) / 60

    def save_error_map(self, data, output):
        lab = SitkReader(os.path.join(data, self.file_names['label']))
        seg = SitkReader(os.path.join(output, self.file_names['segmentation']))
        lab_array = lab.to_numpy()
        seg_array = seg.to_numpy()

        seg_array[seg_array == lab_array] = 0

        # save label map
        seg.to_image_file(os.path.join(output, 'error_map.img'))
