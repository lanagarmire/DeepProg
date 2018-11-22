"""
Load the 3-omics model used in the original study

tsv files used in the original study are available in the ./data folder of this project.
However, theses files must be decompressed using this function in linux:
gzip -d *.gz.

"""

# Python import needed
from simdeep.extract_data import LoadData
from simdeep.simdeep_analysis import SimDeep
from simdeep.config import PATH_THIS_FILE

from collections import OrderedDict

from os.path import isfile

from sys import exit


def main():
    """ Main function excecuted """
    path_data = PATH_THIS_FILE + "/../data/"

    # Testing if the files were decompressed in the good repository
    try:
        assert(isfile(path_data + "meth.tsv"))
        assert(isfile(path_data + "rna.tsv"))
        assert(isfile(path_data + "mir.tsv"))
    except AssertionError:
        print('gz files in {0} must be decompressed !\n exiting...'.format(path_data))
        exit(1)

    # Tsv files used in the original study in the appropriate order
    tsv_files = OrderedDict([
        ('MIR', 'mir.tsv'),
        ('METH', 'meth.tsv'),
        ('RNA', 'rna.tsv'),
    ])

    # File with survival event
    survival_tsv = 'survival.tsv'

    # As test dataset we will use the rna.tsv only
    tsv_test = 'rna.tsv'
    # because it is the same data, we should use the same survival file
    test_survival = 'survival.tsv'

    # class to load and prepare the data
    dataset = LoadData(
        path_data=path_data,
        training_tsv=tsv_files,
        survival_tsv=survival_tsv,
        tsv_test=tsv_test,
        survival_tsv_test=test_survival
    )

    # Instanciate a SimDeep instance
    simDeep = SimDeep(dataset=dataset)
    # load the training dataset
    simDeep.load_training_dataset()
    # Load the full model
    simDeep.load_encoder('encoder_seed_s0_full.h5')
    # identify nodes linked to survival
    simDeep.look_for_survival_nodes()
    # predict labels of the training set using kmeans
    simDeep.predict_labels()

    # Finally, load test set
    simDeep.load_test_dataset()
    # And predict labels and survivals pvalue using the rna.tsv only
    labels, pvalue = simDeep.predict_labels_on_test_dataset()

    #All the parameters are attributes of the SimDeep instance:
    # simDeep.labels
    # simDeep.test_labels
    # simDeep.test_labels_proba
    # ... etc...


if __name__ == "__main__":
    main()
