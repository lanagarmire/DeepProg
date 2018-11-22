from simdeep.simdeep_analysis import SimDeep


def main():
    """ """
    simDeep = SimDeep()
    simDeep.load_training_dataset()
    simDeep.fit()

    simDeep.load_test_dataset()
    simDeep.predict_labels_on_test_dataset()


if __name__ == "__main__":
    main()
