import cv2
import json
import optparse
import os
import time
import numpy as np
import IPython

MACHINE = 'TRI'
# MACHINE = 'Stanford'

if MACHINE == 'TRI':
  # downsampled from:'/mnt/parallel/stip/ANN_hanh2/20170907_prolix_trial_ANN_hanh2-09-07-2017_15-44-07_idx00.mkv'
  test_video_in = '/mnt/parallel/stip/ANN_hanh2/ANN_hanh2_downsample_fps12.mkv'
  test_video_out = 'ANN_hanh2_fps12_masked.mp4'
  test_annot = '/mnt/parallel/stip/annotation/20170907_prolix_trial_ANN_hanh2-09-07-2017_15-44-07.concat.12fps.mp4.json'
else:
  test_video_in = '/sailhome/bingbin/STR-PIP/datasets/STIP/ANN_hanh2_downsampled.mkv'
  test_video_out = 'ANN_hanh2_fps12_masked.mp4'
  test_annot = '/sailhome/bingbin/STR-PIP/datasets/STIP/annotations/20170907_prolix_trial_ANN_hanh2-09-07-2017_15-44-07.concat.12fps.mp4.json'


def check_color(crossed):
    if crossed:
        return (0, 0, 255)
    return (0, 255, 0)


def create_rect(box):
    x1, y1 = int(box['x1']), int(box['y1'])
    x2, y2 = int(box['x2']), int(box['y2'])

    return x1, y1, x2, y2


def create_writer(capture, options):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FPS))

    print('create_writer:', options.saved_video_path)
    writer = cv2.VideoWriter(options.saved_video_path,
                             fourcc,
                             2,
                             (width, height))

    return writer


def get_params(frame_data):
    boxes = [f['box'] for f in frame_data]
    ids = [f['matchIds'] for f in frame_data]
    crossed = [f['crossed'] for f in frame_data]

    return boxes, ids, crossed


def parse_options():
    parser = optparse.OptionParser()
    parser.add_option('-v', '--video',
                      dest='video_path',
                      default=test_video_in)
    parser.add_option('-j', '--json',
                      dest='json_path',
                      default=test_annot)
    parser.add_option('-u', '--until',
                      type='int',
                      default=None)
    parser.add_option('-w', '--write_video',
                      dest='saved_video_path',
                      default=test_video_out)
    options, remainder = parser.parse_args()

    # Check for errors.
    if options.video_path is None: 
        raise Exception('Undefined video')
    if options.json_path is None:
        raise Exception('Undefined json_file')

    return options


def Main():
    options = parse_options()
    print('here')
    # Open VideoCapture.
    cap = cv2.VideoCapture(options.video_path)

    # Load json file with annotations.
    with open(options.json_path, 'r') as f:
        data = json.load(f)['frames']

    lastKeyFrame = int(list(data.keys())[-1])

    writer = create_writer(cap, options)

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_no = 1
    while True:
        wait_key = 25
        flag, img = cap.read()
        if frame_no % 120 == 0:
            print('Processed {0} frames'.format(frame_no))

        if frame_no % 6 != 0:
            frame_no += 1
            continue

        key = str(int(frame_no / 6 + 1))

        boxes = data.get(key)

        if boxes == None:
            boxes = []

        # Create list of trackers each 60 frames.
        boxes, ids, crossed = get_params(boxes)
        mask = np.zeros(img.shape)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = create_rect(box)
            if ids[i] == '4' or ids[i] == '16':
                print(frame_no, key, box)
                print((x1, y1, x2, y2))

            mask[y1:y2,x1:x2, :] = 1
            # crossed_color = check_color(crossed[i])
            # cv2.rectangle(img, (x1, y1), (x2, y2), crossed_color, 2, 1)
        if '4' in ids or '16' in ids:
            wait_key = 0


        # print('writing')
        writer.write(np.uint8(img*mask))

        if frame_no == options.until:
            break

        if frame_no > lastKeyFrame:
            break

        frame_no += 1

    cap.release()
    writer.release()


if __name__ == '__main__':
    Main()
