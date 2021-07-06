import cv2
import re
import sys
import warnings
import argparse
import requests
from joblib import Parallel, delayed
import os
import json
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torch.multiprocessing as mp


headers = {
    'User-Agent': 'Googlebot-Image/1.0',
    'X-Forwarded-For': '64.18.15.200'
}


def print_segment_line(info=''):
    sys.stderr.flush()
    print((' ' + info.strip() + ' ').center(50, '='), flush=True)


def clean_caption(cap):
    new_cap = cap
    new_cap = new_cap.replace(r'&amp;', ' ').replace(r'quot;', ' ').replace('amp;', ' ')  # remove html tags
    new_cap = re.sub(r'\([^>]+?\)', '', new_cap)  # remove everything in (...)
    new_cap = re.sub(r'\.+', '.', new_cap)  # remove redundant dots
    new_cap = re.sub(r'[^\S\n\t]+', ' ', new_cap)  # remove redundant spacing
    new_cap = new_cap.strip()
    return new_cap


def delete_invalid(index, path):
    image_name = '{:0>7d}.jpg'.format(index)
    image_dir = os.path.join(path, image_name[:4], image_name)

    if not os.path.isfile(image_dir):
        return

    try:
        img = Image.open(image_dir)
        img.verify()
        assert img.size[0] > 50 and img.size[1] > 50
    except (IOError, ValueError, AssertionError) as e:
        os.remove(image_dir)
        print('Deleted corrupt image:', image_dir, flush=True)


def download_image(index, url, path):
    image_name = '{:0>7d}.jpg'.format(index)
    image_dir = os.path.join(path, image_name[:4], image_name)

    if not os.path.isdir(os.path.join(path, image_name[:4])):
        try:
            os.mkdir(os.path.join(path, image_name[:4]))
        except (FileExistsError) as e:
            print("Multi process conflict by creating {}, but it's fine.".format(image_dir))

    if os.path.isfile(image_dir):
        return

    try:
        response = requests.get(url, stream=False, timeout=5, allow_redirects=True, headers=headers)
        with open(image_dir, 'wb') as file:
            response.raw.decode_content = True
            file.write(response.content)
    except Exception as e:
        print('failed to download {}'.format(url), flush=True)


def build_index(index, caption, data_dir):
    image_name = '{:0>7d}.jpg'.format(index)
    image_file = os.path.join(data_dir, image_name[:4], image_name)
    img = cv2.imread(image_file)

    if img is not None:  # check if image is valid
        return (os.path.join(data_dir, image_name[:4], image_name), caption)

    return None


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                        help='Download images')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='download/load images from this directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='CC annotation directory with '
                             '"Train_GCC-training.tsv", and "Validation_GCC-1.1.0-Validation.tsv"')
    parser.add_argument('--max_index', type=int, default=-1,
                        help='The maximum index')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='number of jobs for downloading')
    parser.add_argument('--delete_invalid', action='store_true',
                        help='Delete invalid images in data_dir')
    args = parser.parse_args()

    if args.download and args.data_dir is None:
        raise ValueError('if --download is set, --data_dir must be specified')

    with open(os.path.join(args.annot_dir, 'Train_GCC-training.tsv')) as f:
        train_file = [list(map(lambda x: x.strip(), line.split('\t'))) for line in f.readlines()]

    with open(os.path.join(args.annot_dir, 'Validation_GCC-1.1.0-Validation.tsv')) as f:
        val_file = [list(map(lambda x: x.strip(), line.split('\t'))) for line in f.readlines()]

    split_dict = {'images_train': train_file, 'images_val': val_file}

    # make directory for splits
    for split in split_dict.keys():
        path = os.path.join(args.data_dir, split)
        if not os.path.isdir(path):
            os.mkdir(path)

    start = datetime.now()
    for split, data_file in split_dict.items():
        path = os.path.join(args.data_dir, split)
        urls = [x[1] for x in data_file]

        if args.download:
            # download images
            Parallel(n_jobs=args.n_jobs)(
                delayed(download_image)(index, url, path)
                for index, url in enumerate(tqdm(urls[:args.max_index]))
            )

        if args.delete_invalid:
            # delete invalid images
            Parallel(n_jobs=args.n_jobs)(
                delayed(delete_invalid)(index, path)
                for index in tqdm(range(len(urls[:args.max_index])))
            )

    # build index of valid images
    train_captions = [x[0] for x in train_file]
    train_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, os.path.join(args.data_dir, 'images_train'))
        for index, caption in enumerate(tqdm(train_captions[:args.max_index]))
    )
    train_data = list(filter(lambda x: x is not None, train_data))

    val_captions = [x[0] for x in val_file]
    val_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, os.path.join(args.data_dir, 'images_val'))
        for index, caption in enumerate(tqdm(val_captions[:args.max_index]))
    )
    val_data = list(filter(lambda x: x is not None, val_data))

    json.dump(train_data, open(os.path.join(args.data_dir, 'train_annot.json'), 'w'))
    json.dump(val_data, open(os.path.join(args.data_dir, 'val_annot.json'), 'w'))
    
    print_segment_line("Build index complete in: " + str(datetime.now() - start))

    make_arrow(args.data_dir, '../../Datasets/ViLT')

    print_segment_line("Build ViLT complete!")
