import warnings
import numpy as np


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
          isfactor=False):
    """
    input:
        :values: array    values of activities
        :isdead: array <binary>    is dead?
        :nbdays: array <int>
    return:
        pvalues from wald test
fitOrg <- survfit(formula = Surv(new_df$days_relapse,new_df$event_relapse) ~ new_df$SNF)
plot(fitOrg,col=2:(7+1),xlab="Years",ylab="Probablity of survival",lwd=3,main=paste0("survival analysis ","(Log-rank P value ",p.val,")"), mark.time = TRUE)

    """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    values_str = 'values'

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

    return pvalue

def surv_mean(isdead,nbdays):
    """ """
    isdead = FloatVector(isdead)
    nbdays = FloatVector(nbdays)

    surv = rob.r.summary(survival.Surv(nbdays, isdead))
    return float(surv[3].split(':')[1])
