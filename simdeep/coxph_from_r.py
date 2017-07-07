import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from rpy2 import robjects as rob
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector
    from rpy2.robjects import StrVector
    from rpy2.robjects import Formula

    survival = importr('survival')


def coxph(values,
          isdead,
          nbdays,
          do_KM_plot=False,
          png_path='./',
          fig_name='KM_plot.png',
          isfactor=False):
    """
    input:
        :values: array    values of activities
        :isdead: array <binary>    is dead?
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

    res = survival.coxph(cox)

    pvalue = rob.r.summary(res)[-5][2]

    surv = survival.survfit(cox)

    # color = ['green', 'blue', 'red']

    if do_KM_plot:
        rob.r.png("{0}/{1}.png".format(png_path, fig_name.replace('.png', '')))
        rob.r.plot(surv,
                   col=rob.r("2:8"),
                   xlab="Days",
                   ylab="Probablity of survival",
                   sub='pvalue: {0}'.format(pvalue),
                   lwd=3,
                   mark_time=True
            )

        rob.r("dev.off()")
        print("{0}/{1}.png saved!".format(png_path, fig_name.replace('.png', '')))

    return pvalue

def surv_mean(isdead,nbdays):
    """ """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    surv = rob.r.summary(survival.Surv(nbdays, isdead))
    return float(surv[3].split(':')[1])
