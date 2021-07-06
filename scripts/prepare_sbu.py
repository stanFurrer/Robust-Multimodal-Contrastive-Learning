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

from vilt.utils.write_sbu import make_arrow


headers = {
    'User-Agent': 'Googlebot-Image/1.0',
    'X-Forwarded-For': '64.18.15.200'
}


def print_segment_line(info=''):
    sys.stderr.flush()
    print((' ' + info.strip() + ' ').center(50, '='), flush=True)


def delete_invalid(index, path):
    image_name = '{:0>7d}.jpg'.format(index)
    image_dir = os.path.join(path, image_name[:4], image_name)

    if not os.path.isfile(image_dir):
        return

    try:
        img = Image.open(image_dir)
        img.verify()
        assert img.size[0] > 10 and img.size[1] > 10
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
    except:
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
                        help='SBU annotation directory with '
                             '"SBU_captioned_photo_dataset_captions.txt", and "SBU_captioned_photo_dataset_urls.txt"')
    parser.add_argument('--max_index', type=int, default=-1,
                        help='The maximum index')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='number of jobs for downloading')
    parser.add_argument('--delete_invalid', action='store_true',
                        help='Delete invalid images in data_dir')
    args = parser.parse_args()

    if args.download and args.data_dir is None:
        raise ValueError('if --download is set, --data_dir must be specified')

    with open(os.path.join(args.annot_dir, 'SBU_captioned_photo_dataset_captions.txt')) as f:
        captions = f.readlines()

    with open(os.path.join(args.annot_dir, 'SBU_captioned_photo_dataset_urls.txt')) as f:
        urls = f.readlines()

    start = datetime.now()
    if args.download:
        # download images
        Parallel(n_jobs=args.n_jobs)(
            delayed(download_image)(index, url, os.path.join(args.data_dir, 'images_train'))
            for index, url in enumerate(tqdm(urls[:args.max_index]))
        )

    if args.delete_invalid:
        # delete invalid images
        Parallel(n_jobs=args.n_jobs)(
            delayed(delete_invalid)(index, os.path.join(args.data_dir, 'images_train'))
            for index in tqdm(range(len(urls[:args.max_index])))
        )
        print_segment_line("Download complete in: " + str(datetime.now() - start))

    start = datetime.now()

    # build index of valid images
    raw_data = Parallel(n_jobs=args.n_jobs)(
        delayed(build_index)(index, caption, os.path.join(args.data_dir, 'images_train'))
        for index, caption in enumerate(tqdm(captions[:args.max_index]))
    )

    raw_data = list(filter(lambda x: x is not None, raw_data))

    json.dump(raw_data, open(os.path.join(args.data_dir, 'annot.json'), 'w'))
    
    print_segment_line("Build index complete in: " + str(datetime.now() - start))

    make_arrow(args.data_dir, '../../Datasets/ViLT')

    print_segment_line("Build ViLT complete!")
