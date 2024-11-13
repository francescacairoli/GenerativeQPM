# GenQPM: Conformal Dynamic-aware Quantitative Predictive Monitoring for Multi-Modal Scenarios

**Abstract**
We consider the problem of quantitative predictive monitoring (QPM), i.e., predicting at runtime the level of satisfaction of a desired property from the current state of the system. QPM methods support stochastic processes and rich specifications given in Signal Temporal Logic (STL). QPM methods need to be efficient to enable timely interventions against predicted violations while providing correctness guarantees. QPM derives prediction intervals that are highly efficient to compute and with probabilistic guarantees, in that the intervals cover with desired probability the STL robustness values relative to the stochastic evolution of the system. 
Existing QPMs fail to provide useful results in scenarios showing multi-modal dynamics, where some modes may present higher satisfaction than others and are thus preferable options.  State-of-the-art QPMs result in intervals covering a wide range of robustness values providing little practical information over the mode-specific level of satisfaction. In this work, we leverage deep generative models to capture the stochastic dynamics of the system and propose a dynamic-aware QPM, referred to as GenQPM. Moreover, we tailor the conformal inference framework to provide mode-conditional guarantees. In practice, GenQPM provides bounds over robustness values for each dynamical mode, representing for instance the possible choices for an agent interacting with a multi-modal environment.
We demonstrate the effectiveness of GenQPM on multiple autonomous driving case studies.


**Training of the generative model**
Names are `signal`, `crossroad`, `navigation`
`python exec_csdi.py --model_name NAME --nepochs NEPOCHS`


**Run GenQPM**
To load precomputed results `--load True` and `--calload True`

`python exec_gen_cqr_sol1.py --model_name NAME --modelfolder GEN_ID --property_idx P_ID --classifier FLAG`

`FLAG = False` if the mode predictor is exact, `FLAG = True` if the mode predictor is learnt from data.

For multi-agent case study:
`python exec_gen_cqr_sol1multi.py --model_name crossroad --modelfolder 467 --property_idx 3 --classifier True`


Baseline
`python exec_gen_cqr_base.py --model_name NAME --modelfolder GEN_ID --property_idx P_ID --classifier FLAG`
