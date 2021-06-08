import ray
from simdeep.simdeep_analysis import SimDeep

from simdeep.config import NB_CLUSTERS
from simdeep.config import CLUSTER_ARRAY
from simdeep.config import PVALUE_THRESHOLD
from simdeep.config import CINDEX_THRESHOLD
from simdeep.config import CLASSIFIER_TYPE

from simdeep.config import MIXTURE_PARAMS
from simdeep.config import PATH_RESULTS
from simdeep.config import PROJECT_NAME
from simdeep.config import CLASSIFICATION_METHOD

from simdeep.config import CLUSTER_EVAL_METHOD
from simdeep.config import CLUSTER_METHOD
from simdeep.config import NB_THREADS_COXPH
from simdeep.config import NB_SELECTED_FEATURES
from simdeep.config import LOAD_EXISTING_MODELS
from simdeep.config import NODES_SELECTION
from simdeep.config import PATH_TO_SAVE_MODEL

from simdeep.config import USE_AUTOENCODERS
from simdeep.config import FEATURE_SURV_ANALYSIS
from simdeep.config import CLUSTERING_OMICS
from simdeep.config import USE_R_PACKAGES_FOR_SURVIVAL


@ray.remote
class SimDeepDistributed(SimDeep):
    def __init__(
            self,
            nb_clusters=NB_CLUSTERS,
            pvalue_thres=PVALUE_THRESHOLD,
            cindex_thres=CINDEX_THRESHOLD,
            use_autoencoders=USE_AUTOENCODERS,
            feature_surv_analysis=FEATURE_SURV_ANALYSIS,
            cluster_method=CLUSTER_METHOD,
            cluster_eval_method=CLUSTER_EVAL_METHOD,
            classifier_type=CLASSIFIER_TYPE,
            project_name=PROJECT_NAME,
            path_results=PATH_RESULTS,
            cluster_array=CLUSTER_ARRAY,
            nb_selected_features=NB_SELECTED_FEATURES,
            mixture_params=MIXTURE_PARAMS,
            node_selection=NODES_SELECTION,
            nb_threads_coxph=NB_THREADS_COXPH,
            classification_method=CLASSIFICATION_METHOD,
            load_existing_models=LOAD_EXISTING_MODELS,
            clustering_omics=CLUSTERING_OMICS,
            path_to_save_model=PATH_TO_SAVE_MODEL,
            metadata_usage=None,
            feature_selection_usage='individual',
            use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL,
            alternative_embedding=None,
            kwargs_alternative_embedding={},
            do_KM_plot=True,
            verbose=True,
            _isboosting=False,
            dataset=None,
            deep_model_additional_args={}):
        """
        """
        SimDeep.__init__(
            self,
            nb_clusters=nb_clusters,
            pvalue_thres=pvalue_thres,
            cindex_thres=cindex_thres,
            cluster_method=cluster_method,
            cluster_eval_method=cluster_eval_method,
            classifier_type=classifier_type,
            project_name=project_name,
            clustering_omics=clustering_omics,
            path_results=path_results,
            use_autoencoders=use_autoencoders,
            feature_surv_analysis=feature_surv_analysis,
            cluster_array=cluster_array,
            nb_selected_features=nb_selected_features,
            mixture_params=mixture_params,
            node_selection=node_selection,
            nb_threads_coxph=nb_threads_coxph,
            feature_selection_usage=feature_selection_usage,
            metadata_usage=metadata_usage,
            classification_method=classification_method,
            load_existing_models=load_existing_models,
            path_to_save_model=path_to_save_model,
            do_KM_plot=do_KM_plot,
            alternative_embedding=alternative_embedding,
            kwargs_alternative_embedding=kwargs_alternative_embedding,
            verbose=verbose,
            use_r_packages=use_r_packages,
            _isboosting=_isboosting,
            dataset=dataset,
            deep_model_additional_args=deep_model_additional_args)
