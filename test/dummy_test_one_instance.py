"""
test one instance of SimDeep
"""

from os.path import abspath
from os.path import split

from os.path import isfile
from os.path import isdir

from os import remove
from shutil import rmtree



def test_instance():
    """
    test one instance of SimDeep
    """
    from simdeep.simdeep_analysis import SimDeep
    from simdeep.extract_data import LoadData

    PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

    TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv', 'MIR': 'mir_dummy.tsv'}
    SURVIVAL_TSV = 'survival_dummy.tsv'

    PROJECT_NAME = 'TestProject'
    NB_EPOCH = 3

    dataset = LoadData(path_data=PATH_DATA,
                   survival_tsv=SURVIVAL_TSV,
                   training_tsv=TRAINING_TSV)

    simdeep = SimDeep(dataset=dataset,
                      project_name=PROJECT_NAME,
                      path_results=PATH_DATA,
                      nb_epoch=NB_EPOCH,
                      seed=4)
    simdeep.load_training_dataset()
    simdeep.fit()
    simdeep.predict_labels_on_full_dataset()
    simdeep.predict_labels_on_test_fold()

    simdeep.load_new_test_dataset(
        {'RNA': 'rna_test_dummy.tsv'},
        'survival_test_dummy.tsv',
        'dummy')

    simdeep.predict_labels_on_test_dataset()

    from glob import glob

    for fil in glob('{0}/{1}*'.format(PATH_DATA, PROJECT_NAME)):
        if isfile(fil):
            remove(fil)
        elif isdir(fil):
            rmtree(fil)


if __name__ == '__main__':
    test_instance()
