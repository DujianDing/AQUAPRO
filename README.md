# On Efficient Approximate Queries over Machine Learning Models

This is the codebase for [On Efficient Approximate Queries over Machine Learning Models](https://www.vldb.org/pvldb/vol16/p918-ding.pdf).

# Requirements

You will need the following installed:

* python>=3.5
* numpy
* pandas
* numba
* pickle
* scipy
* seaborn

To complete the baseline comparison w.r.t. SUPG, you will need to `cd supg/` and follow the installation instructions.

# Reproduce Experiment Results

Run `aquapro.py`. This will generate all experiment results including baseline comparison under `./results`. The whole process can take a few hours to complete. 

Run `aquapro_figure_generator.py` to generate figures used in the paper. You need to manually change the experiment name `exp_name` inside `aquapro_figure_generator.py`. The mapping is as follows,

* Figure 7: `exp_name='PQA'`
* Figure 8: `exp_name='COMP_topk_PQA'`
* Figure 9: `exp_name='COMP_PQA_PQE'`
* Figure 10: `exp_name='CSC'`
* Figure 11 (left): `exp_name='overhead'`
* Figure 11 (right): `exp_name='COMP_CSC_CSE'`
* Figure 12 & 13: `exp_name='CMPR'`
* Figure 14: `exp_name='scalability'`
