import sys, os
sys.path.append('..')
import utils


def get_user_ids(line):
    # return left uid, right uid
    return eval(line.split('->')[0])[0], eval(line.split('->')[1])[0]

def main():
    year = 16
    root_folder = os.path.join('..', '..')
    tree_folder = os.path.join(root_folder, 'dataset', 'twitter', 'twitter' + str(year), 'tree')
    tree_files = utils.ls(tree_folder)
    print('number of tree files: {}'.format(len(tree_files)))
    user_ids = []
    i = 1
    for tree_file in tree_files:
        print(i)
        i += 1
        lines = utils.read_lines(os.path.join(tree_folder, tree_file))
        source_user_id = get_user_ids(lines[0])[1]
        user_ids.append(source_user_id)
        for line in lines:
            left_user_id, right_user_id = get_user_ids(line)
            if left_user_id == source_user_id:
                user_ids.append(right_user_id)
    print('number of users: {}'.format(len(user_ids)))
    utils.save(user_ids, os.path.join(root_folder, 'temp', 'user_ids_twitter_' + str(year) + '.txt'))

if __name__ == '__main__':
    main()
