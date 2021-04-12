# Tutorial: use DeepProg from the docker image

We created a docker image with deepprog python dependencies installed. The docker image (`opoirion/deepprog_docker:v1`) can be downloaded using `docker pull` and used to analyse a multi-omic dataset.

## Installation with docker
Docker needs to be installed first

```bash
docker pull opoirion/deepprog_docker:v1
```

The version of the package installed correspond to the versions described in the `requirements_tested.txt`. Thus, they are NOT the most up to date python packages, especially regarding the `ray` installed package (installed version is 0.8.4). Since ray is used to configure the nodes, memories, CPUs when distributing DeepProg in a cluster, the API to use might differ with the most up-to-date ray API.

## Usage
the docker cntainer needs to have access to three folders:
1. the input folder containing the matrices and the survival data
2. the output folder where will be generated the output file
3. the folder containing the DeepProg python code to launch

Then, the DeepProg docker can be invoked using the following command

```bash
 docker run \
     -v <ABSOLUTE PATH FOR INPUT DATA>:/input \
      -v <ABSOLUTE PATH FOR OUTPUT DATA>:/output \
      -v <ABSOLUTE PATH FOR THE SCRIPT>:/code \
      --rm \ # remove the container once the computation is finished
      --name greedy_beaver \ # Name of the temporary docker process to create
      deepprog_docker \ # name of the DeepProg docker image to invoke
      python3.8 /code/<NAME OF THE PYTHON SCRIPT FILE>
```


## Example
1. Create three folders for input, output, and scripts
```bash
cd $HOME
mkdir local_input
mkdir local_output
mkdir local_code
```

2. Go to `local_input` and download the matrices and survival data from the STAD cancer here `http://ns102669.ip-147-135-37.us/DeepProg/matrices/STAD/`

```bash
cd local_input
wget http://ns102669.ip-147-135-37.us/DeepProg/matrices/STAD/meth_mapped_STAD.tsv
wget http://ns102669.ip-147-135-37.us/DeepProg/matrices/STAD/mir_mapped_STAD.tsv
wget http://ns102669.ip-147-135-37.us/DeepProg/matrices/STAD/rna_mapped_STAD.tsv
wget http://ns102669.ip-147-135-37.us/DeepProg/matrices/STAD/surv_mapped_STAD.tsv
```

3. Go to `local_code` and open a text editor to create the following script named `processing_STAD.py`

```python
### script: processing_STAD.py

# Import DeepProg class
from simdeep.simdeep_boosting import SimDeepBoosting

# Defining global variables for input and output paths the mounted folder from the docker image
PATH_DATA = '/input/' # virtual folder
PATH_RESULTS = '/output/' # virtual folder


# Defining a main function
def main():
    """
    processing of STAD multiomic cancer
    """

    #Downloaded matrix files
    TRAINING_TSV = {
        'RNA': 'rna_mapped_STAD.tsv',
        'METH': 'meth_mapped_STAD.tsv',
        'MIR': 'mir_mapped_STAD.tsv'
        }

    #survival file
    SURVIVAL_TSV = 'surv_mapped_STAD.tsv'
    
    # survival flag
    survival_flag = {'patient_id': ‘SampleID', 'survival': ‘time’,'event': ‘event’}

    # output folder name
    OUTPUT_NAME = 'STAD_docker'
    PROJECT_NAME = 'STAD_docker'

    # Import ray, the library that will distribute our model computation accros different nodes
    import ray

    ray.init(
        webui_host='127.0.0.1', # This option is required when using ray from the docker image
        num_cpus=10 #
        )

    # Random seed defining how the input dataset will be split
    SEED = 3
    # Number of DeepProg submodels to create
    nb_it = 10
    EPOCHS = 10

    boosting = SimDeepBoosting(
        nb_it=nb_it,
        split_n_fold=3,
        survival_flag=survival_flag,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_RESULTS,
        epochs=EPOCHS,
        distribute=True, # Option to use ray cluster scheduler
        seed=SEED)

    # Fit the model
    boosting.fit()
    # Save the labels of each submodels
    boosting.save_models_classes()
    boosting.save_cv_models_classes()

    # Predict labels on the full (trainings + cv splits) datasets
    boosting.predict_labels_on_full_dataset()

    # Compute consistency
    boosting.compute_clusters_consistency_for_full_labels()
    # Performance indexes
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    # Feature scores
    boosting.compute_feature_scores_per_cluster()
    boosting.collect_number_of_features_per_omic()

    boosting.write_feature_score_per_cluster()

    # Close clusters and free memory
    ray.shutdown()


# Excecute main function if this file is launched as a script
if __name__ == '__main__':
    main()
```

4. After saving this script, we are now ready to launch DeepProg using the docker image:

```bash
 docker run \
     -v ~/local_input:/input \
      -v ~/local_output:/output \
      -v ~/local_code:/code \
      --rm \
      --name greedy_beaver \
      deepprog_docker \
      python3.8 /code/processing_STAD.py
```

5. After the execution, a new output folder inside `~/local_output` should have been created

```bash
ls ~/local_output/'STAD_docker

# Output
-rw-r--r-- 1 root root  22K Mar 30 07:37 STAD_docker_KM_plot_boosting_full.pdf
-rw-r--r-- 1 root root 830K Mar 30 07:37 STAD_docker_features_anticorrelated_scores_per_clusters.tsv
-rw-r--r-- 1 root root 812K Mar 30 07:37 STAD_docker_features_scores_per_clusters.tsv
-rw-r--r-- 1 root root  16K Mar 30 07:37 STAD_docker_full_labels.tsv
-rw-r--r-- 1 root root  22K Mar 30 07:37 STAD_docker_proba_KM_plot_boosting_full.pdf
drwxr-xr-x 2 root root 4.0K Mar 30 07:37 saved_models_classes
drwxr-xr-x 2 root root 4.0K Mar 30 07:37 saved_models_cv_classes

```

6. the same methodology should be followed for adding more analyses, such as predicting a test dataset, embedding, or perform a hyperparameter tuning. Also, a better description of DeepProg different options is available in the other section of this tutorial
