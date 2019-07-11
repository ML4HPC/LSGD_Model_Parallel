#LSGD_Model_Parallel

07/09/2019

- model_parallel.py: the class defination of model-parallel and pipeline-model-parallel

- m_LSGD.py: a modification based on LSGD. m_LSGD will import the class defination in model_parallel.py

- run_m_LSGD.sh: the run script on Cori-GPUs.


***********

Currently model- and pipeline-model- parallel are only supported for ResNet50 architecture. Only 2 GPUs are used for 1 model-parallel.



