import warnings

from numpy import nan

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from simdeep.config import USE_R_PACKAGES_FOR_SURVIVAL

import matplotlib
matplotlib.use('Agg')

import pylab as plt

import pandas as pd

FloatVector = None
StrVector = None
Formula = None
survival = None
rob = None
survcomp = None
glmnet = None

NALogicalType = type(None)


if USE_R_PACKAGES_FOR_SURVIVAL:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from rpy2 import robjects as rob
        from rpy2.robjects.packages import importr
        from rpy2.robjects import FloatVector
        from rpy2.robjects import StrVector
        from rpy2.robjects import Formula

        survival = importr('survival')
        survcomp = importr('survcomp')
        glmnet = importr('glmnet')

        try:
            from rpy2.rinterface import NALogicalType
        except Exception:
            from rpy2.rinterface_lib.na_values import NALogicalType


import numpy as np

def main():
    """
    DEBUG
    """
    isdead = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
    nbdays = [24, 10, 25, 50, 14, 10 ,100, 10, 50, 10]
    values = [0, 1, 1, 0 , 1, 2, 0, 1, 0, 0]
    np.random.seed(2016)

    pvalue = coxph_from_python(values, isdead, nbdays, do_KM_plot=True)
    cindex = c_index_from_python(
        values, isdead, nbdays, values, isdead, nbdays)

    values_proba = np.random.random(10)
    pvalue_proba = coxph(values_proba, isdead, nbdays,
                         do_KM_plot=False,
                         dichotomize_afterward=False)
    print(pvalue_proba)

    # matrix = np.random.random((10, 2))

    values_test = np.random.randint(0, 1, 10)

    pvalue = coxph(values, isdead, nbdays, isfactor=True)
    print('pvalue:', pvalue)
    # surv = coxph(Mr, 1, nbdays, isdead)

    cindex = c_index(
        values,
        isdead,
        nbdays,
        values_test,
        isdead,
        nbdays)

    print('c index:', cindex)
    matrix = np.random.random((10, 5))
    matrix_test = np.random.random((10, 5))

    cindex = c_index_multiple(
        matrix,
        isdead,
        nbdays,
        matrix_test,
        isdead,
        nbdays)

    print('c index:', cindex)

    print('surv med: {0}'.format(surv_median(isdead, nbdays)))
    print('surv mean: {0}'.format(surv_mean(isdead, nbdays)))

def coxph_from_python(
        values,
        isdead,
        nbdays,
        do_KM_plot=False,
        png_path='./',
        metadata_mat=None,
        dichotomize_afterward=False,
        fig_name='KM_plot.pdf',
        penalizer=0.01,
        l1_ratio=0.0,
        isfactor=False):
    """
    """
    values = np.asarray(values)
    isdead = np.asarray(isdead)
    nbdays = np.asarray(nbdays)

    if isfactor:
        values = np.asarray(values).astype("str")

    if metadata_mat is not None:
        frame = {
            "values": values,
            "isdead": isdead,
            "nbdays": nbdays
        }

        for key in metadata_mat:
            frame[key] = metadata_mat[key]

        frame = pd.DataFrame(frame)

    else:
        frame = pd.DataFrame({
            "values": values,
            "isdead": isdead,
            "nbdays": nbdays
        })
        penalizer = 0.0

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(frame, "nbdays", "isdead")

        except Exception:
            return np.nan

    pvalue = cph.log_likelihood_ratio_test().p_value
    cindex = cph.concordance_index_

    if do_KM_plot:
        fig, ax = plt.subplots(figsize=(10, 10))

        kaplan = KaplanMeierFitter()

        for label in set(values):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kaplan.fit(
                    #values[values==label],
                nbdays[values==label],
                    event_observed=isdead[values==label],
                    label='cluster nb. {0}'.format(label)
                )

            kaplan.plot(ax=ax,
                        ci_alpha=0.15)

        ax.set_xlabel('time unit')
        ax.set_title('pval.: {0: .1e} CI: {1: .2f}'.format(
            pvalue, cindex),
                     fontsize=16,
                     fontweight='bold')

        figname = "{0}/{1}.pdf".format(
            png_path, fig_name.replace('.pdf', '').replace('.png', ''))

        fig.savefig(figname)
        print('Figure saved in: {0}'.format(figname))

    return pvalue

def coxph(values,
          isdead,
          nbdays,
          do_KM_plot=False,
          metadata_mat=None,
          png_path='./',
          dichotomize_afterward=False,
          fig_name='KM_plot.png',
          isfactor=False,
          use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL,
          seed=None,
):
    """
    """
    if seed:
        np.random.seed(int(seed))

    if use_r_packages:
        func = coxph_from_r
    else:
        func = coxph_from_python

    return func(
        values,
        isdead,
        nbdays,
        do_KM_plot=do_KM_plot,
        png_path=png_path,
        dichotomize_afterward=dichotomize_afterward,
        fig_name=fig_name,
        metadata_mat=metadata_mat,
        isfactor=isfactor
    )

def coxph_from_r(
        values,
        isdead,
        nbdays,
        do_KM_plot=False,
        metadata_mat=None,
        png_path='./',
        dichotomize_afterward=False,
        fig_name='KM_plot.png',
        isfactor=False):
    """
    input:
        :values: array    values of activities
        :isdead: array <binary>    Event occured int boolean: 0/1
        :nbdays: array <int>
    return:
        pvalues from wald test
    """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    if isfactor:
        # values_str = 'factor({0})'.format(values_str)
        values = StrVector(values)
    else:
        values = FloatVector(values)

    cox = Formula('Surv(nbdays, isdead) ~ values')

    cox.environment['nbdays'] = nbdays
    cox.environment['isdead'] = isdead
    cox.environment['values'] = values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = survival.coxph(cox)
    except Exception as e:
        warnings.warn('Cox-PH didnt fit return NaN: {0}'.format(e))
        return np.nan

    pvalue = rob.r.summary(res)[-5][2]
    # color = ['green', 'blue', 'red']
    pvalue_to_print = pvalue

    if do_KM_plot:
        if dichotomize_afterward:
            frame = rob.r('data.frame')
            predicted = np.array(rob.r.predict(res, frame(values=values)))
            new_values = predicted.copy()
            med = np.median(predicted)
            new_values[predicted >= med] = 0
            new_values[predicted < med] = 1
            new_values = FloatVector(new_values)
            pvalue_to_print = coxph(new_values, isdead, nbdays)

            cox.environment['values'] = new_values

        surv = survival.survfit(cox)
        rob.r.png("{0}/{1}.png".format(png_path, fig_name.replace('.png', '')))
        rob.r.plot(surv,
                   col=rob.r("2:8"),
                   xlab="Days",
                   ylab="Probablity of survival",
                   sub='pvalue: {0}'.format(pvalue_to_print),
                   lwd=3,
                   mark_time=True
            )

        rob.r("dev.off()")
        print("{0}/{1}.png saved!".format(png_path, fig_name.replace('.png', '')))

        del res, surv, cox

    return pvalue

def c_index(
        values,
        isdead,
        nbdays,
        values_test,
        isdead_test,
        nbdays_test,
        isfactor=False,
        use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL,
        seed=None,
        ):
    """
    """
    if seed:
        np.random.seed(int(seed))

    if use_r_packages:
        func = c_index_from_r
    else:
        func = c_index_from_python

    return func(
        values,
        isdead,
        nbdays,
        values_test,
        isdead_test,
        nbdays_test,
        isfactor=isfactor
    )


def c_index_multiple(
        values,
        isdead,
        nbdays,
        values_test,
        isdead_test,
        nbdays_test,
        isfactor=False,
        use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL,
        seed=None,
        ):
    """
    """
    if seed:
        np.random.seed(int(seed))

    if use_r_packages:
        func = c_index_multiple_from_r
    else:
        func = c_index_multiple_from_python

    return func(
        values,
        isdead,
        nbdays,
        values_test,
        isdead_test,
        nbdays_test,
        isfactor=isfactor
    )

def c_index_from_python(
        values,
        isdead,
        nbdays,
        values_test,
        isdead_test,
        nbdays_test,
        isfactor=False):
    """
    """

    if isfactor:
        values = np.asarray(values).astype("str")
        values_test = np.asarray(values_test).astype("str")

    frame = pd.DataFrame({
        "values": values,
        "isdead": isdead,
        "nbdays": nbdays
    })

    frame_test = pd.DataFrame({
        "values": values_test,
        "isdead": isdead_test,
        "nbdays": nbdays_test
    })

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(frame, "nbdays", "isdead")
    except Exception as e:
        print(e)
        return np.nan

    cindex = cph.score(frame_test,
                       scoring_method="concordance_index")

    return cindex


def c_index_multiple_from_python(
        matrix,
        isdead,
        nbdays,
        matrix_test,
        isdead_test,
        nbdays_test,
        isfactor=False):
    """
    """
    frame = pd.DataFrame(matrix)
    frame["isdead"] = isdead
    frame["nbdays"] = nbdays

    frame_test = pd.DataFrame(matrix_test)
    frame_test["isdead"] = isdead_test
    frame_test["nbdays"] = nbdays_test

    cph = CoxPHFitter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(frame, "nbdays", "isdead")
    except Exception as e:
        print(e)
        return np.nan

    cindex = cph.score(frame_test,
                       scoring_method="concordance_index")

    return cindex


def c_index_from_r(values,
            isdead,
            nbdays,
            values_test,
            isdead_test,
            nbdays_test,
            isfactor=False):
    """ """
    rob.r('set.seed(2016)')
    isdead = FloatVector(isdead)
    isdead_test = FloatVector(isdead_test)

    nbdays = FloatVector(nbdays)
    nbdays_test = FloatVector(nbdays_test)

    values = FloatVector(values)
    values_test = FloatVector(values_test)

    if isfactor:
        values = StrVector(values)
        values_test = StrVector(values_test)

    cox = Formula('Surv(nbdays, isdead) ~ values')

    cox.environment['nbdays'] = nbdays
    cox.environment['isdead'] = isdead
    cox.environment['values'] = values

    res = survival.coxph(cox)
    frame = rob.r('data.frame')
    predict = rob.r.predict(res, frame(values=values_test))
    concordance_index = rob.r('concordance.index')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_index = concordance_index(predict,
                                        nbdays_test,
                                        isdead_test,
            method='noether')
    except Exception as e:
        print("exception found for c index!: {0}".format(e))
        return nan

    del res, cox, frame

    return c_index[0][0]

def c_index_multiple_from_r(
        matrix,
        isdead,
        nbdays,
        matrix_test,
        isdead_test,
        nbdays_test,
        lambda_val=None):
    """
    """
    rob.r('set.seed(2016)')

    if matrix.shape[1] < 2:
        return np.nan

    nbdays[nbdays == 0] = 1
    nbdays_test[nbdays_test == 0] = 1

    isdead = FloatVector(isdead)
    isdead_test = FloatVector(isdead_test)

    nbdays = FloatVector(nbdays)
    nbdays_test = FloatVector(nbdays_test)

    matrix = convert_to_rmatrix(matrix)
    matrix_test = convert_to_rmatrix(matrix_test)

    surv = survival.Surv(nbdays, isdead)

    cv_glmnet = rob.r('cv.glmnet')
    glmnet = rob.r('glmnet')

    arg = {'lambda': lambda_val}

    if not lambda_val:
        cv_fit = cv_glmnet(matrix, surv, family='cox', alpha=0)
        arg = {'lambda': min(cv_fit[0])}

    fit = glmnet(matrix, surv, family='cox', alpha=0, **arg)

    predict = rob.r.predict(fit, matrix_test)
    concordance_index = rob.r('concordance.index')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_index = concordance_index(predict,
                                        nbdays_test,
                                        isdead_test,
                                        method='noether')
    except Exception as e:
        print("exception found for c index multiple!: {0}".format(e))
        return None

    return c_index[0][0]


def predict_with_coxph_glmnet(
        matrix,
        isdead,
        nbdays,
        matrix_test,
        alpha=0.5,
        lambda_val=None):
    """
    """
    rob.r('set.seed(2020)')

    if matrix.shape[1] < 2:
        return np.nan

    nbdays[nbdays == 0] = 1

    isdead = FloatVector(isdead)

    nbdays = FloatVector(nbdays)

    matrix = convert_to_rmatrix(matrix)
    matrix_test = convert_to_rmatrix(matrix_test)

    surv = survival.Surv(nbdays, isdead)

    cv_glmnet = rob.r('cv.glmnet')
    glmnet = rob.r('glmnet')

    arg = {'lambda': lambda_val}

    if not lambda_val:
        cv_fit = cv_glmnet(matrix, surv, family='cox',
                           alpha=alpha)
        arg = {'lambda': min(cv_fit[0])}

    fit = glmnet(matrix, surv, family='cox', alpha=0, **arg)

    return np.asarray(rob.r.predict(fit, matrix_test)).T[0]


def convert_to_rmatrix(data):
    """ """
    shape = data.shape
    return rob.r.t(
        rob.r.matrix(
            rob.FloatVector(
                list(np.resize(data, shape[0] * shape[1]))),
            nrow=shape[1], ncol=shape[0])
    )


def surv_mean(isdead, nbdays,
              use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL):
    """
    """
    if use_r_packages:
        func = surv_mean_from_r
    else:
        func = surv_mean_from_python

    return func(isdead, nbdays)

def surv_median(
        isdead, nbdays,
        use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL):
    """
    """
    if use_r_packages:
        func = surv_median_from_r
    else:
        func = surv_median_from_python

    return func(isdead, nbdays)

def surv_mean_from_python(isdead,nbdays):
    """
    """
    from lifelines.utils import restricted_mean_survival_time

    kaplan = KaplanMeierFitter()

    kaplan.fit(
        nbdays,
        event_observed=isdead,
    )

    survmean = restricted_mean_survival_time(kaplan)

    return survmean

def surv_median_from_python(isdead,nbdays):
    """
    """
    kaplan = KaplanMeierFitter()
    np.random.seed(2020)
    kaplan.fit(
        nbdays,
        event_observed=isdead,
    )

    return kaplan.median_survival_time_

def surv_mean_from_r(isdead,nbdays):
    """ """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    surv = rob.r.summary(survival.Surv(nbdays, isdead))

    return float(surv[3].split(':')[1])

def surv_median_from_r(isdead,nbdays):
    """ """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    surv = rob.r.summary(survival.Surv(nbdays, isdead))

    return float(surv[2].split(':')[1])

if __name__ == "__main__":
    main()
