import sys, os
sys.path.append('..')
import utils

import json, time, datetime, numpy
project_folder = os.path.join('..', '..')

def get_user(user_id, dbc):
    sql = '''SELECT json FROM user where id = ?'''
    tup = (user_id, )
    cursor = dbc.cursor()
    rs = cursor.execute(sql, tup).fetchall()
    if rs:
        return json.loads(rs[0][0])
    else:
        return None

def get_user_feature(user):
    f = []
    f.append(len(user['name'])) 
    f.append(len(user['screen_name']))
    f.append(len(user['description']))

    f.append(user['followers_count'])
    f.append(user['friends_count'])
    f.append(user['listed_count'])
    f.append(user['favourites_count'])
    f.append(user['statuses_count'])

    time_now = 'Thu Jan 11 00:00:00 +0000 2018'
    ts_now = utils.str_to_timestamp(time_now)
    ts_user = utils.str_to_timestamp(user['created_at'])             
    f.append( int((ts_now - ts_user) /(3600*24)) )

    f.append(int(bool(user['url'])))
    f.append(int(user['protected']))
    f.append(int(user['geo_enabled']))
    f.append(int(user['verified']))

    f.append(int(user['profile_use_background_image']))
    f.append(int(user['has_extended_profile']))
    f.append(int(user['default_profile']))
    f.append(int(user['default_profile_image']))

    return f

def get_user_ids(line):
    # return left uid, right uid
    return eval(line.split('->')[0])[0], eval(line.split('->')[1])[0]

def get_retweet_path(tweet_id, year, path_len):
    user_ids = []
    tree_file = os.path.join(project_folder, 'dataset', 'twitter', 'twitter' + str(year), 'tree', tweet_id + '.txt')
    lines = utils.read_lines(tree_file)
    source_user_id = get_user_ids(lines[0])[1]
    user_ids.append(source_user_id)
    for line in lines[1 : path_len]:
        left_user_id, right_user_id = get_user_ids(line)
        if left_user_id == source_user_id:
            user_ids.append(right_user_id)
    return user_ids

def get_feature_vec(tweet_id, year, dbc, path_len):
    user_ids = get_retweet_path(tweet_id, year, path_len)
    f_vec = []
    for user_id in user_ids:
        if get_user(user_id, dbc):
            f_vec.append(get_user_feature(get_user(user_id, dbc)))
    return f_vec

def get_label_dict(label_file):
    d = {}
    lines = utils.read_lines(label_file)
    for line in lines:
        label = line.split(':')[0]
        tweet_id = line.split(':')[1].strip()
        d[tweet_id] = label
    return d

def main():
    year = 16
    path_len = 100
    dbc = utils.db_connect(os.path.join(project_folder, 'db', 'twitter_user.db'))
    label_file = os.path.join(project_folder, 'dataset', 'twitter', 'twitter' + str(year), 'label.txt')
    label_dict = get_label_dict(label_file)
    nb_feature = len(get_user_feature(get_user('33164207', dbc)))
    
    y = []
    x = []
    c = 0
    for tweet_id in label_dict:
        print(c)
        c += 1
        if label_dict[tweet_id] == 'false':
            y.append(1)
        elif label_dict[tweet_id] == 'non-rumor':
            y.append(0)
        else:
            continue
        f_vec = get_feature_vec(tweet_id, year, dbc, path_len)
        if len(f_vec) < path_len:
            for i in range(path_len - len(f_vec)):
                f_vec.append([0 for j in range(nb_feature)])
        x.append(f_vec)
    
    y = numpy.array(y)
    x = numpy.array(x)

    print(x.shape, y.shape)
    numpy.save(os.path.join(project_folder, 'feature', 'twitter' + str(year), 'x.npy'), x)
    numpy.save(os.path.join(project_folder, 'feature', 'twitter' + str(year), 'y.npy'), y)

if __name__ == '__main__':
    main()