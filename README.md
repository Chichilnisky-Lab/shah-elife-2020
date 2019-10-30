# shah-elife-2019
Code for "Nishal P. Shah, N. Brackbill, C. Rhoades, A. Tikidji-Hamburyan, G. Goetz, A. Litke, A. Sher, E. P. Simoncelli, E.J. Chichilnisky. Inference of Nonlinear Receptive Field Subunits with Spike-Triggered Clustering. ELife, 2019"

Prepare data: 
```python
stim_use  # stimulus, numpy array, size: number of samples x number of pixels
resp_use  # binned cell response, numpy array, size: number of samples x number of cells 
nsub  # number of subunits (int)
# TODO(bhaishahster): add other details
```

Fit the model: 

```python
op = su_model.spike_triggered_clustering(stim_use, resp_use, nsub,
                                         tms_train,
                                         tms_validate,
                                         steps_max=10000, eps=1e-9,
                                         projection_type=projection_type,
                                         neighbor_mat=neighbor_mat,
                                         lam_proj=lam_proj, eps_proj=0.01,
                                         save_filename_partial=save_filename_partial, 
                                         fitting_phases=[0, 1])

k, b, nl_params, lam_log_train, lam_log_validation, fitting_phase, fit_params = op
```


Evaluate loss on test data:

```python
fitting_phase = 1
k, b, nl_params = fit_params[fitting_phase] 
lam_test = su_model.compute_fr_loss(k, b, stim_use[tms_test, :], resp_use[tms_test, :],
                                    nl_params=nl_params)
```

