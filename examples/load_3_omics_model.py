"""
Load the 3-omics model

tsv files used in the original study are available in the ./data folder of this project.
However, theses files must be decompressed using this function in linux:
gzip -d *.gz.

"""

from simdeep.extract_data import LoadData
from simdeep.simdeep_analysis import SimDeep
from simdeep.config import PATH_THIS_FILE

from os.path import isfile


def main():
    """ """
    path_data = PATH_THIS_FILE + "/../data/"

    # Testing if the files were decompressed in the good repository
    assert(isfile(path_data + "meth.tsv"))
    assert(isfile(path_data + "rna.tsv"))
    assert(isfile(path_data + "mir.tsv"))


if __name__ == "__main__":
    main()
