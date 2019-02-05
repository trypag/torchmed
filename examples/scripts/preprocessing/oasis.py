import argparse
import math
import os
import time
import shutil

from torchmed.utils.multiproc import parallelize_system_calls
from torchmed.readers import SitkReader

from mappings import OASISMapping


parser = argparse.ArgumentParser(
    description='Dataset extractor and pre-processing for the OASIS dataset')
parser.add_argument('data', metavar='DIR', help='path to the oasis dataset')
parser.add_argument('output', metavar='DIR', help='path to output dataset')
parser.add_argument('-n', '--nb-workers', default=2, type=int, metavar='N',
                    help='Number of workers for the parallel execution of Affine Registration')
parser.add_argument('--data-split-ratio', default=0.70, type=float, metavar='N',
                    help='Pourcentage of train data vs validation data')


def main():
    args = parser.parse_args()
    mapping = OASISMapping()

    print('\n' +
          '\n',
          ('##### Automatic Segmentation of brain MRI #####') + '\n',
          ('#####      By Pierre-Antoine Ganaye       #####') + '\n',
          ('#####         CREATIS Laboratory          #####'),
          '\n' * 2,
          ('The dataset can be downloaded at https://www.oasis-brains.org/#data') + '\n',
          ('This script will preprocess the OASIS-1 T1-w brain MRI dataset') + '\n',
          ('-------------------------------------------------------------') + '\n')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    os.makedirs(os.path.join(args.output, 'train'))
    os.makedirs(os.path.join(args.output, 'validation'))

    # only work with scans from session 1 "OAS1"
    sessions = [session for session in os.listdir(args.data) if session.startswith('OAS1')]
    filtered_sessions = [s for s in sessions if s[5:9] not in mapping.avoid_patients]
    filtered_sessions = sorted(filtered_sessions)

    # allowed data
    command_list = []
    command_list2 = []

    print(('1/ --> Reading train and validation directories'))
    print(('2/ --> Creating corresponding tree hierarchy'))
    print(('3/ --> Copying files to destination folders'))
    for patient_session in filtered_sessions:
        # create destination dirs
        output_dir = os.path.join(args.output, patient_session)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get name of the image, pick first scan of each session
        session_files = os.path.join(args.data, patient_session)
        raw_files = os.path.join(session_files, 'RAW')
        scan = [scan for scan in os.listdir(raw_files) if scan.endswith('_mpr-1_anon.img')]

        # write input and destination to file
        input_scan = os.path.join(raw_files, scan[0])
        registered_scan = os.path.join(output_dir, 'im_mni.nii.gz')
        registered_scan_bc = os.path.join(output_dir, 'im_mni_bc.nii.gz')
        aff_trans = os.path.join(output_dir, 'mni_aff_transf.mat')

        # affine registration with flirt
        command = ('flirt -searchrx -180 180 -searchry -180 180'
                   ' -searchrz -180 180 -in {} -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz '
                   '-interp trilinear -o {} -omat {}').format(
            input_scan, registered_scan, aff_trans
        )
        command_list.append(command)

        # bias field correction
        command = ('N4BiasFieldCorrection -d 3 -i {}'
                   ' -o {} -s 4 -b 200 -d 3').format(
            registered_scan, registered_scan_bc)
        command_list2.append(command)

    print(('4/ --> Affine registration to MNI space and bias field correction'))
    print('Can take several hours depending on the number of workers ({})'.format(
        args.nb_workers
    ))
    parallelize_system_calls(args.nb_workers, command_list)
    time.sleep(30)
    parallelize_system_calls(args.nb_workers // 2, command_list2)
    time.sleep(30)

    print('5/ --> Mean centering and reduction')
    # estimate mean and variance on the first 50 images
    n_mean_estimate = 50
    mean, var = (0, 0)
    for patient_session in filtered_sessions[0:n_mean_estimate]:
        output_dir = os.path.join(args.output, patient_session)
        registered_scan_bc = os.path.join(output_dir, 'im_mni_bc.nii.gz')

        brain = SitkReader(registered_scan_bc, torch_type='torch.FloatTensor').to_torch()
        mean += brain.mean()
        var += brain.var()

    mean = mean / n_mean_estimate
    var = var / n_mean_estimate

    for patient_session in filtered_sessions:
        output_dir = os.path.join(args.output, patient_session)
        registered_scan_bc = os.path.join(output_dir, 'im_mni_bc.nii.gz')
        registered_scan_norm = os.path.join(output_dir, 'prepro_im_mni_bc.nii.gz')

        brain = SitkReader(registered_scan_bc, torch_type='torch.FloatTensor')
        brain_array = brain.to_torch() \
                           .add(-mean) \
                           .div(math.sqrt(var))

        brain.to_image_file(registered_scan_norm)

    train_dir = os.path.join(args.output, 'train')
    val_dir = os.path.join(args.output, 'validation')

    stats_log = open(os.path.join(train_dir, 'stats_log.txt'), "w")
    stats_log.write('average mean: {:.10f}\n'
                    'average standard deviation: {:.10f}'.format(mean, math.sqrt(var)))
    stats_log.flush()

    # split into train validation test
    train_patient_number = math.floor(args.data_split_ratio * len(filtered_sessions))
    train_patients = filtered_sessions[:train_patient_number]
    validation_patients = filtered_sessions[train_patient_number:]

    allowed_train = open(os.path.join(train_dir, 'allowed_data.txt'), "w")
    allowed_val = open(os.path.join(val_dir, 'allowed_data.txt'), "w")

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


if __name__ == '__main__':
    main()
