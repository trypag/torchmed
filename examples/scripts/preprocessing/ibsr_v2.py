import argparse
import cv2
import math
import matplotlib.pyplot
import numpy as np
import os
import shutil
import SimpleITK as sitk
import time
import torch

from mappings import IbsrMapping
from torchmed.utils.multiproc import parallelize_system_calls
from torchmed.readers import SitkReader

parser = argparse.ArgumentParser(
    description='Dataset extractor and pre-processing for the IBSRv2 dataset')
parser.add_argument('data', metavar='DIR', help='path to the IBSRv2 dataset')
parser.add_argument('output', metavar='DIR', help='path to output dataset')
parser.add_argument('-n', '--nb-workers', default=2, type=int, metavar='N',
                    help='Number of workers for the parallel execution of Affine Registration')


def main():
    args = parser.parse_args()
    mapping = IbsrMapping()
    train_size = 10
    val_size = 3

    print('\n' +
          '\n',
          ('##### Automatic Segmentation of brain MRI #####') + '\n',
          ('#####      By Pierre-Antoine Ganaye       #####') + '\n',
          ('#####         CREATIS Laboratory          #####'),
          '\n' * 2,
          ('The dataset can be downloaded at https://www.nitrc.org/frs/?group_id=48') + '\n',
          ('This script will preprocess the IBSRv2 T1-w brain MRI dataset') + '\n',
          ('-------------------------------------------------------------') + '\n')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    os.makedirs(os.path.join(args.output, 'validation'))
    os.makedirs(os.path.join(args.output, 'test'))
    os.makedirs(os.path.join(args.output, 'train'))

    sessions = [session for session in os.listdir(args.data) if session.startswith('IBSR')]
    filtered_sessions = sorted(sessions)
    command_list = []
    command_list2 = []
    command_list3 = []

    print(('1/ --> Reading train and validation directories'))
    print(('2/ --> Creating corresponding tree hierarchy'))
    print(('3/ --> Copying files to destination folders'))
    labels = set()
    for patient_session in filtered_sessions:
        # create destination dirs
        output_dir = os.path.join(args.output, patient_session)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get name of the image, pick first scan of each session
        session_files = os.path.join(args.data, patient_session)

        # remove uncompressed image, otherwise flirt goes crazy
        if os.path.exists(os.path.join(session_files, patient_session + '_ana.nii')):
            os.remove(os.path.join(session_files, patient_session + '_ana.nii'))

        # write input and destination to file
        input_scan = os.path.join(session_files, patient_session + '_ana.nii.gz')
        input_seg = os.path.join(session_files, patient_session + '_seg_ana.nii.gz')
        registered_scan = os.path.join(output_dir, 'im_mni_bc.nii.gz')
        transform = os.path.join(output_dir, 'mni_aff_transf.mat')
        transform_c3d = os.path.join(output_dir, 'mni_aff_transf.c3dmat')
        seg = os.path.join(output_dir, 'seg_mni.nii.gz')

        x = SitkReader(input_seg).to_torch()
        labels = labels.union(set(np.unique(x.numpy())))

        # affine registration with flirt
        command = ('flirt -searchrx -180 180 -searchry -180 180'
                   ' -searchrz -180 180 -in {} -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz '
                   '-interp trilinear -o {} -omat {}').format(
            input_scan, registered_scan, transform)
        command_list.append(command)

        command = ('c3d_affine_tool -ref {} -src {} {} -fsl2ras -o {}').format(
            registered_scan, input_scan, transform, transform_c3d)
        command_list2.append(command)

        command = ('c3d {} -popas ref {} -split '
                   '-foreach -insert ref 1 -reslice-matrix {} '
                   '-endfor -merge -o {}').format(
            registered_scan, input_seg, transform_c3d, seg)
        command_list3.append(command)

    label_file = open(os.path.join(args.output, 'labels.txt'), "w")
    label_file.write(str(labels))
    label_file.flush()

    print(('4/ --> Affine registration to MNI space and segmentation resampling'))
    print('Can take several hours depending on the number of workers ({})'.format(
        args.nb_workers))
    parallelize_system_calls(args.nb_workers, command_list)
    time.sleep(30)
    parallelize_system_calls(args.nb_workers, command_list2)
    time.sleep(10)
    parallelize_system_calls(2, command_list3)

    print('5/ --> Mean centering and reduction')
    mean, var = (0, 0)
    # eval mean and var on half on the dataset
    for patient_session in filtered_sessions[0:train_size]:
        output_dir = os.path.join(args.output, patient_session)
        registered_scan_bc = os.path.join(output_dir, 'im_mni_bc.nii.gz')

        brain = SitkReader(registered_scan_bc, torch_type='torch.FloatTensor').to_torch()
        mean += brain.mean()
        var += brain.var()

    mean = mean / train_size
    var = var / train_size

    for patient_session in filtered_sessions:
        output_dir = os.path.join(args.output, patient_session)
        registered_scan_bc = os.path.join(output_dir, 'im_mni_bc.nii.gz')
        registered_scan_norm = os.path.join(output_dir, 'prepro_im_mni_bc.nii.gz')

        brain = SitkReader(registered_scan_bc, torch_type='torch.FloatTensor')
        brain_array = brain.to_torch() \
                           .add(-mean) \
                           .div(math.sqrt(var))

        brain.to_image_file(registered_scan_norm, sitk.sitkFloat32)

    print('6/ --> Remapping the labels')
    mapping = IbsrMapping()
    invalid_classes = set()
    for i in mapping.all_labels:
        if i in mapping.ignore_labels:
            invalid_classes.add(i)

    for patient_session in filtered_sessions:
        output_dir = os.path.join(args.output, patient_session)
        seg = os.path.join(output_dir, 'seg_mni.nii.gz')

        label = SitkReader(seg)
        label_array = label.to_numpy()

        # filter invalid class
        for inv_class in invalid_classes:
            label_array[label_array == inv_class] = -1

        # remap valid class
        for class_id in mapping.overall_labels:
            label_array[label_array == class_id] = mapping[class_id]

        # write back the changes to file
        label.to_image_file(output_dir + '/prepro_seg_mni.nii.gz')

    train_dir = os.path.join(args.output, 'train')
    val_dir = os.path.join(args.output, 'validation')
    test_dir = os.path.join(args.output, 'test')

    # split into train validation test
    train_patients = filtered_sessions[:train_size]
    validation_patients = filtered_sessions[train_size:train_size + val_size]
    test_patients = filtered_sessions[train_size + val_size:]

    allowed_train = open(os.path.join(train_dir, 'allowed_data.txt'), "w")
    allowed_val = open(os.path.join(val_dir, 'allowed_data.txt'), "w")
    allowed_test = open(os.path.join(test_dir, 'allowed_data.txt'), "w")

    # move train
    for patient_session in train_patients:
        output_dir = os.path.join(args.output, patient_session)
        dest_dir = os.path.join(train_dir, patient_session)
        allowed_train.write(patient_session + '\n')
        shutil.move(output_dir, dest_dir)
    allowed_train.flush()

    # move validation
    for patient_session in validation_patients:
        output_dir = os.path.join(args.output, patient_session)
        dest_dir = os.path.join(val_dir, patient_session)
        allowed_val.write(patient_session + '\n')
        shutil.move(output_dir, dest_dir)
    allowed_val.flush()

    # move test
    for patient_session in test_patients:
        output_dir = os.path.join(args.output, patient_session)
        dest_dir = os.path.join(test_dir, patient_session)
        allowed_test.write(patient_session + '\n')
        shutil.move(output_dir, dest_dir)
    allowed_test.flush()

    # class statistics on the train dataset
    print('7/ --> Computing class statistics')
    sum_by_class = [0] * mapping.nb_classes
    class_log = open(os.path.join(train_dir, 'class_log.csv'), 'a')
    class_log.write('class;volume\n')
    for train_id in train_patients:
        patient_dir = os.path.join(train_dir, train_id)
        label = SitkReader(patient_dir + '/prepro_seg_mni.nii.gz')
        label_array = label.to_numpy()

        for class_id in range(0, mapping.nb_classes):
            remapped_labels = (label_array == class_id)
            nb_point = np.sum(remapped_labels)
            sum_by_class[class_id] += nb_point

    for class_id in range(0, mapping.nb_classes):
        class_log.write('{};{}\n'.format(class_id, sum_by_class[class_id]))
    class_log.flush()

    stats_log = open(os.path.join(train_dir, 'stats_log.txt'), "w")
    stats_log.write('average mean: {:.10f}\n'
                    'average standard deviation: {:.10f}'.format(mean, math.sqrt(var)))
    stats_log.flush()


if __name__ == '__main__':
    main()
