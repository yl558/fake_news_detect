import sys, os, time
sys.path.append('..')
import utils

import requests
from requests_oauthlib import OAuth1Session, OAuth1

def user_exists(user_id, dbc):
    sql = 'select * from user where id = ?'
    tup = (user_id, )
    cursor = dbc.cursor()
    cursor.execute(sql, tup)
    rs = cursor.fetchall()
    if rs:
        return True
    else:
        return False

def insert_user(user_id, json, dbc):
    sql = 'insert or replace into user values (?, ?)'
    tup = (user_id, json)
    cursor = dbc.cursor()
    cursor.execute(sql, tup)

def select_crawled_user_ids(dbc):
    sql = 'select id from user'
    cursor = dbc.cursor()
    cursor.execute(sql)
    rs = cursor.fetchall()
    return rs

def main():
    year = 15
    root_folder = os.path.join('..', '..')
    user_ids_file = os.path.join(root_folder, 'temp', 'user_ids_twitter_' + str(year) + '.txt')
    user_ids = utils.read(user_ids_file)
    print('# user ids: {}'.format(len(user_ids)))
    dbc = utils.db_connect(os.path.join(root_folder, 'db', 'twitter_user.db'))

    rs = select_crawled_user_ids(dbc)
    crawled_user_ids = []
    for r in rs:
        crawled_user_ids.append(r[0])
    user_ids_not_exist = list(set(user_ids) - set(crawled_user_ids))
    print('# user ids not crawled: {}'.format(len(user_ids_not_exist)))

    client_key= 'ACE4GN7LOv1C5isH1e18VRmeD'
    client_secret='zgKhzxxMph36Bt6jnIrr7ysfvnEaujdmr0pS6jpezbmkWxhIuX'
    resource_owner_key='917033232587214848-18iztS2rsFjIOUg1OrmnOR8kQxz9xIk'
    resource_owner_secret='soS2qjaMG1fgsyM3afh8wnSns0ki4Cu5BIG2uSwkcXfzB'
    oauth = OAuth1(client_key, client_secret, resource_owner_key, resource_owner_secret)
    url = 'https://api.twitter.com/1.1/users/lookup.json'

    user_ids = user_ids_not_exist
    for i in range(int(len(user_ids) / 100) + 1):
        if i < 0:
            continue
        print(i)
        user_id_list = user_ids[i * 100: i * 100 + 100]
        user_ids_str = ''
        for user_id in user_id_list[0:-2]:
            user_ids_str += user_id + ','
        user_ids_str += user_id_list[-1]
        data = {'user_id': user_ids_str}
        r = requests.post(url, data = data, auth = oauth)
        time.sleep(1)
        if r.status_code == 404:
            continue
        res_data = r.json()
        for user_data in res_data:
            user_id = user_data['id_str']
            json_text = json.dumps(res_data)
            insert_user(user_id, json_text, dbc)
        dbc.commit()
    dbc.commit()

if __name__ == '__main__':
    main()