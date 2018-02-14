import sys, os
sys.path.append('..')
import utils

import json, time, datetime, numpy
project_folder = os.path.join('..', '..')

def get_feature(post):
    f = []
    f.append(post['followers_count'])
    f.append(post['bi_followers_count'])
    f.append(post['friends_count'])
    f.append(post['statuses_count'])
    f.append(post['favourites_count'])
    f.append(post['attitudes_count'])

    f.append(len(post['user_description']))
    f.append(len(post['screen_name']))
    f.append(len(post['username']))

    f.append((post['t'] - post['user_created_at']) / (3600 * 24))
    
    f.append(int(post['verified']))
    f.append(int(post['user_geo_enabled']))

    if post['gender'] == 'm':
        f.append(0)
    else:
        f.append(1)

    if post['city'] is None:
        f.append(0)
    else:
        f.append(1)

    if post['province'] is None:
        f.append(0)
    else:
        f.append(1)

    if post['user_location'] is None:
        f.append(0)
    else:
        f.append(1)
    
    return f

def main():
    path_len = 100
    nb_feature = 16
    weibo_file = os.path.join(project_folder, 'dataset', 'weibo', 'weibo.txt')
    lines = utils.read_lines(weibo_file)
    x = []
    y = []
    i = 1
    for line in lines:
        print(i)
        i += 1
        line = line.replace('\t', ' ')
        sp = line.split(' ')
        eid = sp[0].split(':')[1]
        label = sp[1].split(':')[1]
        y.append(int(label))
        f = []
        json_file = os.path.join(project_folder, 'dataset', 'weibo', 'Weibo', eid + '.json')
        text_content = utils.read(json_file)
        json_content = json.loads(text_content)
        for post in json_content[0:path_len]:
            f.append(get_feature(post))
        if len(f) < path_len:
            for j in range(path_len - len(f)):
                f.append([0 for j in range(nb_feature)])
        x.append(f)

    y = numpy.array(y)
    x = numpy.array(x)

    print(x.shape, y.shape)
    numpy.save(os.path.join(project_folder, 'feature', 'weibo', 'x.npy'), x)
    numpy.save(os.path.join(project_folder, 'feature', 'weibo', 'y.npy'), y)

if __name__ == '__main__':
    main()