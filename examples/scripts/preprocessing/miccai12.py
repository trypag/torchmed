import argparse
import math
import os
import shutil
import torch
import time

import torchmed.utils.file as files
from torchmed.readers import SitkReader
from torchmed.utils.multiproc import parallelize_system_calls

from mappings import MiccaiMapping

parser = argparse.ArgumentParser(
    description='Dataset pre-processing for the MICCAI 2012 multi-atlas brain segmentation challenge.')
parser.add_argument('data', metavar='SOURCE_DIR', help='path to the original MICCAI12 dataset')
parser.add_argument('output', metavar='DESTINATION_DIR', help='path to the destination directory')
parser.add_argument('-n', '--nb-workers', default=1, type=int, metavar='N',
                    help='Number of workers')
parser.add_argument('--data-split-ratio', default=0.70, type=float, metavar='N',
                    help='Pourcentage of train data vs validation data')
parser.add_argument('--per-patient-norm', action='store_true',
                    help='normalize images with per patient mean/std (default: train set mean/std)')


def main():
    args = parser.parse_args()
    assert(args.nb_workers >= 1)

    print('\n' +
          '\n',
          ('##### Automatic Segmentation of brain MRI #####') + '\n',
          ('#####      By Pierre-Antoine Ganaye       #####') + '\n',
          ('#####         CREATIS Laboratory          #####'),
          '\n' * 2,
          ('The dataset can be downloaded at https://docs.google.com/forms/d/e/1FAIpQLSfwkdSt7hWo_tjHUDu2stDsxWTaWyLJIUiS_iapbtKaydEMIw/viewform') + '\n',
          ('This script will pre-process the MICCAI12 T1-w brain MRI multi-atlas segmentation dataset') + '\n',
          ('-------------------------------------------------------------') + '\n')

    root_dir = os.path.join(args.data, '')
    dest_dir = os.path.join(args.output, '')

    # create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    os.makedirs(os.path.join(dest_dir, 'train'))
    os.makedirs(os.path.join(dest_dir, 'test'))
    os.makedirs(os.path.join(dest_dir, 'validation'))

    # get IDs of train and test images from the original dataset
    print('1/ --> Reading train and validation directories')
    train_ids = set()
    for train_patient in os.listdir(os.path.join(root_dir, 'training-images')):
        filename = os.path.basename(train_patient)
        train_ids.add(filename[0:4])

    test_ids = set()
    for test_patient in os.listdir(os.path.join(root_dir, 'testing-images')):
        filename = os.path.basename(test_patient)
        test_ids.add(filename[0:4])

    # create destination directories of matching IDs
    print('2/ --> Creating corresponding tree hierarchy')
    # convert set to list so we can iterate over it
    # sorted so that we split in the same way each time this script is called
    train_ids = sorted(list(train_ids))
    test_ids = sorted(list(test_ids))

    for train_patient in train_ids:
        os.makedirs(os.path.join(dest_dir, 'train/' + train_patient))

    for test_patient in test_ids:
        os.makedirs(os.path.join(dest_dir, 'test/' + test_patient))

    print('3/ --> Copying files to destination folders')
    patient_paths = []
    # copy image for the train set
    for train_patient in train_ids:
        # image
        filename = os.path.join(
            root_dir,
            'training-images/' + train_patient + '_3.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'train/' + train_patient + '/' + 'image.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # segmentation map
        filename = os.path.join(
            root_dir,
            'training-labels/' + train_patient + '_3_glm.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'train/' + train_patient + '/' + 'segmentation.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # keep the path to the patient dir for later use
        patient_paths.append(
            os.path.join(dest_dir, 'train/' + train_patient + '/'))

    # copy image for the test set
    for test_patient in test_ids:
        # image
        filename = os.path.join(
            root_dir,
            'testing-images/' + test_patient + '_3.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'test/' + test_patient + '/' + 'image.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # segmentation map
        filename = os.path.join(
            root_dir,
            'testing-labels/' + test_patient + '_3_glm.nii.gz'
        )
        dest_filename = os.path.join(
            dest_dir,
            'test/' + test_patient + '/' + 'segmentation.nii.gz'
        )
        files.copy_file(filename, dest_filename)

        # keep the path to the patient dir for later use
        patient_paths.append(
            os.path.join(dest_dir, 'test/' + test_patient + '/'))

    print('4/ --> Affine registration to MNI space and bias field correction')
    print('Can take up to one hour depending on the chosen number of workers ({})'.format(
        args.nb_workers
    ))

    command_list = []
    command_list2 = []
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        registered_scan = os.path.join(patient, 'im_mni.nii.gz')
        registered_scan_bc = os.path.join(patient, 'im_mni_bc.nii.gz')
        transform = os.path.join(patient, 'mni_aff_transf.mat')

        # affine registration with flirt
        command = ('flirt -searchrx -180 180 -searchry -180 180'
                   ' -searchrz -180 180 -in {} -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz '
                   '-interp trilinear -o {} -omat {}').format(
            input_scan, registered_scan, transform
        )
        command_list.append(command)

        # bias field correction
        command = ('N4BiasFieldCorrection -d 3 -i {}'
                   ' -o {} -s 4 -b 200').format(
            registered_scan, registered_scan_bc)
        command_list2.append(command)

    parallelize_system_calls(args.nb_workers, command_list)
    time.sleep(20)
    parallelize_system_calls(args.nb_workers // 2, command_list2)
    time.sleep(20)

    print('5/ --> Resampling of segmentation maps')
    print('Can take up to 30 minutes.')
    command_list = []
    command_list2 = []
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        reg_scan = os.path.join(patient, 'im_mni.nii.gz')
        transform = os.path.join(patient, 'mni_aff_transf.mat')
        transform_c3d = os.path.join(patient, 'mni_aff_transf.c3dmat')
        seg = os.path.join(patient, 'segmentation.nii.gz')
        dest_seg = os.path.join(patient, 'seg_mni.nii.gz')

        # resampling of the image
        command = ('c3d_affine_tool -ref {} -src {} {} -fsl2ras -o {}').format(
            reg_scan, input_scan, transform, transform_c3d)
        command_list.append(command)

        # resampling of the segmentation map
        command = ('c3d {} -popas ref {} -split '
                   '-foreach -insert ref 1 -reslice-matrix {} '
                   '-endfor -merge -o {}').format(
            reg_scan, seg, transform_c3d, dest_seg)
        command_list2.append(command)

    # limit nb of workers because c3d uses a lot of memory
    parallelize_system_calls(2, command_list)
    time.sleep(20)
    parallelize_system_calls(2, command_list2)

    # delete all original images and segmentation maps
    time.sleep(20)
    for patient in patient_paths:
        input_scan = os.path.join(patient, 'image.nii.gz')
        seg = os.path.join(patient, 'segmentation.nii.gz')
        os.remove(input_scan)
        os.remove(seg)

    # pre-processing of segmentation maps
    print('6/ --> Remapping the labels')
    mapping = MiccaiMapping()
    invalid_classes = set()
    for i in mapping.all_labels:
        if i in mapping.ignore_labels:
            invalid_classes.add(i)

    for patient_path in patient_paths:
        label = SitkReader(patient_path + 'seg_mni.nii.gz')
        label_array = label.to_numpy()

        # filter out invalid class
        for inv_class in invalid_classes:
            label_array[label_array == inv_class] = -1

        # remap all valid class so that label numbers are contiguous
        for class_id in mapping.overall_labels:
            label_array[label_array == class_id] = mapping[class_id]

        # write back the changes to file
        label.to_image_file(patient_path + 'prepro_seg_mni.nii.gz')

    time.sleep(5)

    # split into train validation test
    train_patient_number = math.floor(args.data_split_ratio * len(train_ids))
    train_patients = train_ids[:train_patient_number]
    validation_patients = train_ids[train_patient_number:]

    for validation_patient in validation_patients:
        source = os.path.join(
            dest_dir,
            'train/' + validation_patient
        )
        destination = os.path.join(
            dest_dir,
            'validation/' + validation_patient
        )
        shutil.move(source, destination)

    # class statistics on the train dataset
    print('7/ --> Computing class statistics')
    sum_by_class = [0] * mapping.nb_classes
    class_log = open(os.path.join(dest_dir, 'train/class_log.csv'), 'a')
    class_log.write('class;volume\n')

    for train_id in train_patients:
        patient_dir = os.path.join(dest_dir, 'train/' + train_id)
        label = SitkReader(patient_dir + '/prepro_seg_mni.nii.gz')
        label_array = label.to_numpy()

        for class_id in range(0, mapping.nb_classes):
            remapped_labels = (label_array == class_id)
            nb_point = remapped_labels.sum()
            sum_by_class[class_id] += nb_point

    for class_id in range(0, mapping.nb_classes):
        class_log.write('{};{}\n'.format(class_id, sum_by_class[class_id]))
    class_log.flush()

    print('8/ --> Mean centering and reduction')
    mean, std = (0, 0)

    for train_id in train_patients:
        patient_dir = os.path.join(dest_dir, 'train/' + train_id)
        brain = SitkReader(patient_dir + '/im_mni_bc.nii.gz',
                           torch_type='torch.FloatTensor').to_torch()
        mean += brain.mean()
        std += brain.std()

    train_mean = mean / len(train_patients)
    train_std = std / len(train_patients)

    # train dataset
    for name, dataset in [('train/', train_patients),
                          ('validation/', validation_patients),
                          ('test/', test_ids)]:
        for p_id in dataset:
            patient_dir = os.path.join(dest_dir, name + train_id)
            brain = SitkReader(patient_dir + '/im_mni_bc.nii.gz',
                               torch_type='torch.FloatTensor')
            brain_array = brain.to_numpy()
            if not args.per_patient_norm:
                brain_array[...] = (brain_array - train_mean) / train_std

            else:
                brain_array[...] = (brain_array - brain_array.mean()) / brain_array.std()

            brain.to_image_file(patient_dir + '/prepro_im_mni_bc.nii.gz')

    stats_log = open(os.path.join(dest_dir, 'train/stats_log.txt'), "w")
    stats_log.write('average mean: {:.10f}\n'
                    'average standard deviation: {:.10f}'.format(train_mean, train_std))
    stats_log.flush()

    # allowed data
    allowed_train = open(os.path.join(dest_dir, 'train/allowed_data.txt'), "w")
    for train_patient in train_patients:
        allowed_train.write(train_patient + '\n')
    allowed_train.flush()

    allowed_val = open(os.path.join(dest_dir, 'validation/allowed_data.txt'), "w")
    for val_patient in validation_patients:
        allowed_val.write(val_patient + '\n')
    allowed_val.flush()

    allowed_test = open(os.path.join(dest_dir, 'test/allowed_data.txt'), "w")
    for test_patient in test_ids:
        allowed_test.write(test_patient + '\n')
    allowed_test.flush()


if __name__ == '__main__':
    main()
