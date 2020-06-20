from simdeep.config import PATH_TO_SAVE_MODEL

from os.path import isfile
from os.path import isdir

from os import mkdir

from sys import version_info

# if version_info > (3, 0, 0):
#     import pickle as cPickle
# else:
#     import cPickle

import dill

from time import time

def save_model(boosting, path_to_save_model=PATH_TO_SAVE_MODEL):
    """ """
    if not isdir(path_to_save_model):
        mkdir(path_to_save_model)

    boosting._convert_logs()

    t = time()

    with open('{0}/{1}.pickle'.format(
            path_to_save_model,
            boosting._project_name), 'wb') as f_pick:
        dill.dump(boosting, f_pick)

    print('model saved in %2.1f s at %s/%s.pickle' % (
        time() - t, path_to_save_model, boosting._project_name))

def load_model(project_name, path_model=PATH_TO_SAVE_MODEL):
    """ """
    t = time()
    project_name = project_name.replace('.pickle', '') + '.pickle'

    assert(isfile('{0}/{1}'.format(path_model, project_name)))

    with open('{0}/{1}'.format(path_model, project_name), 'rb') as f_pick:
        boosting = dill.load(f_pick)

    print('model loaded in %2.1f s' % (time() - t))

    return boosting
