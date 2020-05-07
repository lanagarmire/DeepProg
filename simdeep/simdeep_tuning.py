from simdeep.simdeep_boosting import SimDeepBoosting

import numpy as np

from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.skopt import SkOptSearch

from simdeep.config import PATH_RESULTS
from simdeep.config import PROJECT_NAME

from skopt import Optimizer

import pandas as pd


class SimDeepTuning(object):
    """
    Class to optimize hyper-parameters from simdeep

    Parameters:
        :args_to_optimize: Dict with names of args from SimDeepBoosting to optimise with values as tuple (for range) or list (for list of values)
        :path_results: Result folder path used to save the output files (default PATH_RESULTS)
        :project_name: Name of the project. This name will be used to save the output files and create the output folder (default PROJECT_NAME)

    """
    def __init__(self,
                 args_to_optimize,
                 path_results=PATH_RESULTS,
                 project_name=PROJECT_NAME,
                 **deepProgBaseArgs):
        """
        """

        self.args_to_optimize = args_to_optimize
        self.deepProgBaseArgs = deepProgBaseArgs
        self.project_name = project_name
        self.path_results = path_results

        self.results = pd.DataFrame()


    def _objective_only_training(self, config, reporter):
        """
        """
        for i in range(config["iterations"]):
            # norm = dict(config['normalization'])
            print('# CONFIG: {0}'.format(config))

            args = dict(self.deepProgBaseArgs)
            args.update(config)

            args.pop('iterations')

            boosting = SimDeepBoosting(
                path_results=self.path_results,
                project_name=self.project_name,
                **args
            )

            error = "None"

            try:
                boosting.fit()
            except Exception as e:
                pval, pval_proba = 1.0, 1.0
                test_fold_pvalue = 1.0
                test_fold_cindex = 0.0
                cluster_consistency = 0.0
                sum_log_pval = 0
                error = str(e)
            else:
                pval, pval_proba = boosting.predict_labels_on_full_dataset()
                test_fold_pvalue = boosting.compute_pvalue_for_merged_test_fold()
                test_fold_cindex = np.mean(boosting.collect_cindex_for_test_fold())
                cluster_consistency = np.mean(boosting.compute_clusters_consistency_for_full_labels())
                sum_log_pval = - np.log10(1e-128 + pval) - np.log10(1e-128 + test_fold_pvalue)

            reporter(
                timesteps_total=i,
                simdeep_error=error,
                test_fold_pvalue=test_fold_pvalue,
                test_fold_cindex=test_fold_cindex,
                cluster_consistency=cluster_consistency,
                full_pvalue=pval,
                sum_log_pval=sum_log_pval,
                )

    def fit(self, metric="test_fold_pvalue",
            num_samples=10,
            iterations=1,
            max_concurrent=4,
            timesteps_total=100):
        """
        """

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
            "test_fold_pvalue":"min",
            "test_fold_cindex":"max",
            "cluster_consistency":"max",
            "full_pvalue":"min",
            "sum_log_pval": "max"
        }

        try:
            assert metric in metric_authorized
        except Exception:
               raise('{0} should be in {1}'.format(
                   metric, metric_authorized))

        optimizer_header = self.args_to_optimize.keys()
        optimizer_value = [self.args_to_optimize[key] for key in optimizer_header]
        # optimizer_value += [tuple(norm.items()) for norm in self.normalization]

        optimizer = Optimizer(optimizer_value)

        algo = SkOptSearch(
            optimizer, optimizer_header,
            max_concurrent=max_concurrent,
            metric=metric,
            mode=metric_authorized[metric],
        )

        scheduler = AsyncHyperBandScheduler(
            metric=metric,
            mode=metric_authorized[metric])

        self.results = run(
            self._objective_only_training,
            name="skopt_exp_with_warmstart",
            search_alg=algo,
            scheduler=scheduler,
            **config
        )

        index = ['config/' + key for key in self.args_to_optimize]
        index += metric_authorized.keys()

        print('#### best results obtained with:\n{0}'.format(
            self.results.dataframe()[index]
        ))

    def get_results_table(self):
        """
        """
        return self.results.dataframe()

    def save_results_table(self):
        """
        """
        fname = '{0}/{1}_hyperparameters.tsv'.format(
            self.path_results, self.project_name)

        self.results.dataframe().to_csv(fname)

        print('File :{0} written'.format(fname))
