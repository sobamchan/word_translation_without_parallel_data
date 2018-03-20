import os
import json
import requests
import matplotlib.pyplot as plt


def send_slack_img(imgpath):
    token = os.environ["SLACK_API_TOKEN"]
    with open(imgpath, 'rb') as f:
        files = {'file': f}
        params = {'token': token, 'channels': 'C3XSZFUTV'}
        requests.post(url='https://slack.com/api/files.upload',
                      params=params, files=files)
    return imgpath


def output_progress(progress_path, output_dir, prefix=None):
    progress = []
    with open(progress_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            progress.append(json.loads(l))
    keys = [k for k in list(progress[0].keys()) if k != 'epoch']
    x = list(range(len(progress)))
    outputpaths = []
    for k in keys:
        if prefix:
            fname = '{}_{}.png'.format(prefix, k)
        else:
            fname = '{}.png'.format(k)
        outputpath = os.path.join(output_dir, fname)
        y = [float(p[k]) for p in progress]
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel(k)
        plt.savefig(outputpath)
        outputpaths.append(outputpath)
        plt.clf()
    return outputpaths
