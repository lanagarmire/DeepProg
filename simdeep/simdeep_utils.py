from simdeep.config import PATH_TO_SAVE_MODEL

from os.path import isfile
from os.path import isdir

from sys import version_info

if version_info > (3, 0, 0):
    import pickle as cPickle
else:
    import cPickle

from time import time

def save_model(boosting, path_to_save_model=PATH_TO_SAVE_MODEL):
    """ """
    assert(isdir(path_to_save_model))
    boosting._convert_logs()

    t = time()

    with open('{0}/{1}.pickle'.format(
            path_to_save_model,
            boosting._project_name), 'w') as f_pick:
        cPickle.dump(boosting, f_pick)

    print('model saved in %2.1f s at %s/%s.pickle' % (
        time() - t, path_to_save_model, boosting._project_name))


def load_model(project_name, path_model=PATH_TO_SAVE_MODEL):
    """ """
    t = time()
    project_name = project_name.replace('.pickle', '') + '.pickle'

    assert(isfile('{0}/{1}'.format(path_model, project_name)))

    with open('{0}/{1}'.format(path_model, project_name), 'r') as f_pick:
        boosting = cPickle.load(f_pick)

    print('model loaded in %2.1f s' % (time() - t))

    return boosting


def metadata_usage_type(value):
    """ """
    if value not in {None,
                     False,
                     'labels',
                     'new-features',
                     'test-labels',
                     'all', True}:
        raise Exception(
            "metadata_usage_type: {0} should be from the following choices:" \
            " [None, False, 'labels', 'new-features', 'all', True]" \
            .format(value))

    if value == True:
        return 'all'

    return value

def feature_selection_usage_type(value):
    """ """
    if value not in {'individual',
                     'lasso'}:
        raise Exception(
            "feature_selection_usage_type: {0} should be from the following choices:" \
            " ['individual', 'lasso']" \
            .format(value))

    return value
