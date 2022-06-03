from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.special

import supg.supg.datasource as datasource

# Selectors
from supg.supg.selector import ApproxQuery
from supg.supg.selector import UniformPrecisionSelector
from supg.supg.selector import ImportancePrecisionSelector
from supg.supg.selector import ImportancePrecisionTwoStageSelector
from supg.supg.selector import RecallSelector
from supg.supg.selector import NaiveRecallSelector, NaivePrecisionSelector
from supg.supg.selector import JointSelector
# Samplers
from supg.supg.sampler import Sampler
from supg.supg.sampler import NaiveSampler, ImportanceSampler
from supg.supg.sampler import ImportanceReuseSampler
# Estimators
# from supg.estimator import Estimator
# from supg.estimator import NaiveEstimator
from supg.supg.experiments.trial_runner import TrialRunner
from supg.supg.experiments.exp_dict import experiments




def get_sampler(
        sampler_type: str
):
    if sampler_type == "NaiveSampler":
        return NaiveSampler()
    elif sampler_type == "ReuseSampler":
        return ReuseSampler()
    elif sampler_type == "ImportanceSampler":
        return ImportanceSampler()
    elif sampler_type == 'ImportanceReuseSampler':
        return ImportanceReuseSampler()
    return None

def get_estimator(
        estimator_type: str,
        source: datasource.DataSource
):
    if estimator_type == 'NaiveEstimator':
        return NaiveEstimator()


def get_source(
        name: str,
        drop_p: Optional[float]=None,
        seed: Optional[int]=None,
        alpha: Optional[float]=None,
        beta: Optional[float]=None,
        noise: Optional[float]=None,
):
    if name == "jackson":
        return datasource.get_jackson_source(drop_p, seed)
    elif name == 'imagenet':
        return datasource.get_imagenet_source()
    elif name == 'onto':
        return datasource.get_onto_source()
    elif name == 'tacred':
        return datasource.get_tacred_source()
    elif name == 'beta':
        assert alpha is not None and beta is not None
        return datasource.BetaDataSource(alpha, beta, noise=noise)
    raise NotImplementedError


def get_selector(
        type: str,
        query: ApproxQuery,
        sampler: Sampler,
        source: datasource.DataSource,
        # estimator: Estimator
):
    if type == 'UniformPrecisionSelector':
        return UniformPrecisionSelector(query, source, sampler)
    elif type == 'ImportancePrecisionSelector':
        return ImportancePrecisionSelector(query, source, sampler)
    elif type == 'ImportancePrecisionTwoStageSelector':
        return ImportancePrecisionTwoStageSelector(query, source, sampler)
    elif type == 'UniformRecall':
        return RecallSelector(query, source, sampler, sample_mode="uniform")
    elif type == 'ImportanceRecall':
        return RecallSelector(query, source, sampler, sample_mode="sqrt", verbose=False)
    elif type == 'ImportanceRecallProp':
        return RecallSelector(query, source, sampler, sample_mode="prop", verbose=False)
    elif type == 'NaiveRecallSelector':
        return NaiveRecallSelector(query, source, sampler)
    elif type == 'NaivePrecisionSelector':
        return NaivePrecisionSelector(query, source, sampler)
    elif type == 'ImportanceJoint':
        return JointSelector(query, source, sampler, sample_mode='sqrt')
    elif type == 'ImportanceJointProp':
        return JointSelector(query, source, sampler, sample_mode='prop')
    elif type == 'UniformJoint':
        return JointSelector(query, source, sampler, sample_mode='uniform')
    raise NotImplementedError


def run_experiment(cur_experiment, df):
    # print('Running experiment:', experiment_name)
    df['label'] = df['label'].astype('float32')
    source = datasource.DFDataSource(df)

    sampler = get_sampler(cur_experiment["sampler"])
    query = cur_experiment["query"]
    estimator = get_estimator(cur_experiment['estimator'], source)
    selector = get_selector(
            cur_experiment["selector"], query, sampler, source)

    trial_runner = TrialRunner()
    results_df = trial_runner.run_trials(
        selector=selector,
        query=query,
        sampler=sampler,
        source=source,
        nb_trials=cur_experiment["num_trials"],
    )
    # print(results_df)
    agg_results = results_df.aggregate({
        "precision": ["mean", "sem"],
        "recall": ["mean", "sem"],
        # "trial_idx": ["count"],
        "covered": ["mean", "sem"],
        'na_tau': ['mean', 'sem'],
        # 'size': ['mean', 'sem'],
        # 'nb_true': ['mean', 'sem'],
        # 'nb_sampled': ['mean', 'sem']
    })
    return list(agg_results.loc['mean'])
    # print(agg_results)


def main():
    # run_experiment('jackson_precision_imp')

    # run_experiment('jackson_recall_uniform')
    # run_experiment('jackson_precision_uniform')
    # run_experiment('jackson_precision_imp_two')
    run_experiment('jackson_recall_imp')

    # run_experiment('imagenet_recall_uniform')
    # run_experiment('imagenet_recall_imp')
    # run_experiment('imagenet_precision_uniform')
    # run_experiment('imagenet_precision_imp')
    # run_experiment('beta_recall_imp')
    # run_experiment('beta_recall_uniform')
    # run_experiment('beta_precision_uniform')
    # run_experiment('beta_precision_imp')
    # run_experiment('beta_precision_imp_two')
    # run_experiment('beta_jt_imp')
    # run_experiment('beta_jt_uniform')
    # run_experiment('onto_recall_uniform')
    # run_experiment('onto_recall_imp')
    # run_experiment('onto_precision_uniform')
    # run_experiment('onto_precision_imp')
    # run_experiment('tacred_recall_uniform')
    # run_experiment('tacred_recall_imp')
    # run_experiment('tacred_precision_uniform')
    # run_experiment('tacred_precision_imp')

if __name__ == '__main__':
    main()
