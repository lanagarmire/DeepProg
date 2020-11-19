from simdeep.simdeep_boosting import SimDeepBoosting

import numpy as np

from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.skopt import SkOptSearch

from simdeep.config import PATH_RESULTS
from simdeep.config import PROJECT_NAME

from skopt import Optimizer

import pandas as pd

from tabulate import tabulate

from os.path import isdir
from os import mkdir

from distutils.dir_util import mkpath


class SimDeepTuning(object):
    """
    Class to optimize hyper-parameters from simdeep

    Parameters:
        :args_to_optimize: Dict with names of args from SimDeepBoosting to optimise with values as tuple (for range) or list (for list of values)
        :path_results: Result folder path used to save the output files (default PATH_RESULTS)
        :project_name: Name of the project. This name will be used to save the output files and create the output folder (default PROJECT_NAME)
        :test_datasets: a dictionary of test dataset {'test dataset name': ({'omic type': 'matrix file'}, 'survival file')}. The survival file is mendatory to evaluate the test dataset
        :normalization: dictionary of different normalisation types
        :metadata_tsv_test: dictionary of optional metadata file associated with the test dataset {'test dataset name': 'metadata tsv file'}

    """
    def __init__(self,
                 args_to_optimize,
                 path_results=PATH_RESULTS,
                 project_name=PROJECT_NAME,
                 test_datasets={},
                 normalization={},
                 metadata_tsv_test={},
                 survival_flag_test={},
                 **deepProgBaseArgs):
        """
        """

        self.args_to_optimize = args_to_optimize
        self.deepProgBaseArgs = deepProgBaseArgs
        self.project_name = project_name
        self.path_results = path_results
        self.test_datasets = test_datasets
        self._distribute_deepprog = False
        self.normalization = normalization
        self.survival_flag_test = survival_flag_test
        self.metadata_tsv_test = metadata_tsv_test

        self.results = pd.DataFrame()

        self.path_results = '{0}/{1}'.format(
            path_results, project_name)

        self._path_models = '{0}/{1}/models'.format(
            path_results, project_name)

        if not isdir(self.path_results):
            mkpath(self.path_results)

        if not isdir(self._path_models):
            mkdir(self._path_models)

    def _objective_only_training(self, config, reporter):
        """
        """
        for i in range(config["iterations"]):
            # norm = dict(config['normalization'])
            trial_id = "{0}_it{1}".format(reporter._trial_id, i)
            print("#### Trial ID {0} ########".format(trial_id))
            path_results = "{0}".format(self._path_models)

            print('# CONFIG: {0}'.format(config))

            args = dict(self.deepProgBaseArgs)
            args.update(config)

            args.pop('iterations')

            if 'normalization' in args:
                norm = self.normalization[args['normalization']]
                args['normalization'] = norm

            boosting = SimDeepBoosting(
                path_results=path_results,
                project_name=trial_id,
                distribute=self._distribute_deepprog,
                **args
            )

            log_test_pval = 0.0
            test_cindexes = []
            test_consisentcies = []

            (pval,
             test_fold_pvalue,
             test_fold_cindex,
             cluster_consistency,
             sum_log_pval,
             mix_score, error) = self._return_scores(
                 boosting)

            boosting.save_models_classes()

            try:
                boosting.write_logs()
            except Exception:
                pass

            individual_pvals = {}

            for key in self.test_datasets:
                test_dataset, survival = self.test_datasets[key]
                (log_test_pval,
                 sum_log_pval,
                 mix_score,
                 test_pval) = self._return_scores_test(
                     boosting, key,
                     test_dataset, survival,
                     test_cindexes,
                     test_consisentcies,
                     log_test_pval,
                     sum_log_pval,
                     mix_score,
                     error)
                individual_pvals["test_pval_{0}".format(key)] = test_pval

            reporter(
                timesteps_total=i,
                simdeep_error=error,
                log_test_fold_pvalue=-np.log10(1e-128 + test_fold_pvalue),
                test_fold_cindex=test_fold_cindex,
                cluster_consistency=cluster_consistency,
                full_pvalue=pval,
                log_full_pvalue=-np.log10(1e-128 + pval),
                sum_log_pval=sum_log_pval,
                mix_score=mix_score,
                log_test_pval=log_test_pval,
                test_cindex=np.mean(test_cindexes),
                trial_name=trial_id,
                test_consisentcy=np.mean(test_consisentcies),
                **individual_pvals
            )

    def _return_scores_test(
            self,
            boosting, key,
            test_dataset, survival,
            test_cindexes,
            test_consisentcies,
            log_test_pval,
            sum_log_pval,
            mix_score,
            error):
        """ """
        try:
            if key in self.survival_flag_test:
                survival_flag = self.survival_flag_test[key]
            else:
                survival_flag = None

            if key in self.metadata_tsv_test:
                metadata_file = self.metadata_tsv_test[key]
            else:
                metadata_file = None

            print("#### TUNING: loading new test daaset: TSVS: {0} SURVIVAL: {1}"\
                  .format(test_dataset, survival))

            boosting.load_new_test_dataset(
                tsv_dict=test_dataset, # OMIC file of the second test set.
                path_survival_file=survival, # Survival file of the test set
                fname_key='test_{0}'.format(key), # Name of the second test test
                survival_flag=survival_flag,
                metadata_file=metadata_file,
            )

            test_pval, _ = boosting.predict_labels_on_test_dataset()

        except Exception as e:
            print("Exception when predicting test dataset {0}".format(
                e))
            error += str(e)
            test_pval = 1.0
        else:
            test_cindex = boosting.compute_c_indexes_for_test_dataset()
            test_consisentcy = np.mean(boosting \
                    .compute_clusters_consistency_for_test_labels())

            if np.isnan(test_cindex):
                test_cindex = 0.0

            if np.isnan(test_pval):
                test_pval = 1.0

            if np.isnan(test_consisentcy):
                test_consisentcy = 0.0

            test_cindexes.append(test_cindex)
            test_consisentcies.append(test_consisentcy)

            log_test_pval += -np.log10(1e-128 + test_pval)
            sum_log_pval += log_test_pval
            mix_score *= log_test_pval * test_cindex * \
                test_consisentcy

        return log_test_pval, sum_log_pval, mix_score, test_pval

    def _return_scores(self, boosting):
        """ """
        error = "None"

        try:
            boosting.fit()
        except Exception as e:
            pval, _ = 1.0, 1.0
            test_fold_pvalue = 1.0
            test_fold_cindex = 0.0
            cluster_consistency = 0.0
            sum_log_pval = 0
            mix_score = 0.0
            error = str(e)
        else:
            pval, _ = boosting.predict_labels_on_full_dataset()
            test_fold_pvalue = boosting.\
                compute_pvalue_for_merged_test_fold()
            test_fold_cindex = np.mean(
                boosting.collect_cindex_for_test_fold())
            cluster_consistency = np.mean(
                boosting.compute_clusters_consistency_for_full_labels())

            if np.isnan(pval):
                pval = 1.0

            if np.isnan(test_fold_pvalue):
                test_fold_pvalue = 1.0

            if np.isnan(test_fold_cindex):
                test_fold_cindex = 0.0

            sum_log_pval = - np.log10(1e-128 + pval) - np.log10(
                1e-128 + test_fold_pvalue)
            mix_score = sum_log_pval * cluster_consistency * \
                test_fold_cindex

        return (pval,
                test_fold_pvalue,
                test_fold_cindex,
                cluster_consistency,
                sum_log_pval,
                mix_score, error)

    def fit(self, metric="log_test_fold_pvalue",
            num_samples=10,
            iterations=1,
            max_concurrent=4,
            distribute_deepprog=False,
            timesteps_total=100):
        """
        """
        self._distribute_deepprog = distribute_deepprog

        config = {
            "num_samples": num_samples,
            "config": {
                "iterations": iterations,
            },
            "stop": {
                "timesteps_total": timesteps_total
            },
        }

        metric_authorized = {
            "log_test_fold_pvalue": "max",
            "test_fold_cindex": "max",
            "cluster_consistency": "max",
            "log_full_pvalue": "max",
            "sum_log_pval": "max",
            "log_test_pval": "max",
            "test_cindex": "max",
            "mix_score": "max",
            "test_consisentcy": "max",
        }

        try:
            assert metric in metric_authorized
        except Exception:
            raise(Exception('{0} should be in {1}'.format(
                metric, metric_authorized)))

        optimizer_header = self.args_to_optimize.keys()
        optimizer_value = [
            self.args_to_optimize[key] for key in optimizer_header]
        # optimizer_value += [tuple(norm.items()) for norm in self.normalization]

        optimizer = Optimizer(optimizer_value)

        algo = SkOptSearch(
            optimizer, list(optimizer_header),
            max_concurrent=max_concurrent,
            metric=metric,
            mode=metric_authorized[metric],
        )

        scheduler = AsyncHyperBandScheduler(
            metric=metric,
            mode=metric_authorized[metric])

        self.results = run(
            self._objective_only_training,
            name=self.project_name,
            search_alg=algo,
            scheduler=scheduler,
            **config
        )

        index = ['config/' + key for key in self.args_to_optimize]
        index = ['trial_name'] + index + \
            ["test_pval_{0}".format(key)
             for key in self.test_datasets] + [metric, "full_pvalue"]

        df = self.results.dataframe()[index]

        print('#### best results obtained with:\n{0}'.format(
            tabulate(df, headers='keys', tablefmt='psql')
        ))

        fname = '{0}/{1}_hyperparameter_scores_summary.tsv'.format(
            self.path_results, self.project_name)

        df.to_csv(fname, sep="\t")
        print('File :{0} written'.format(fname))

    def get_results_table(self):
        """
        """
        return self.results.dataframe()

    def save_results_table(self, tag=""):
        """
        """
        if tag:
            tag = "_" + tag

        fname = '{0}/{1}{2}_hyperparameters.tsv'.format(
            self.path_results, self.project_name, tag)

        self.results.dataframe().to_csv(fname, sep="\t")

        print('File :{0} written'.format(fname))
