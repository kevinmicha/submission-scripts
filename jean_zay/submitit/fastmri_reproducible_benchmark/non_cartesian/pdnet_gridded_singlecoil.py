from fastmri_recon.evaluate.scripts.nc_eval import evaluate_gridded_pdnet as eval_fun
from fastmri_recon.training_scripts.nc_train import train_gridded_pdnet as train_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

base_params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
}

# run_ids = [
#     'ncpdnet_singlecoil___radial_compound_mssim_dcomp_1610872636',
#     'ncpdnet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
# ]

eval_results = train_eval_grid(
    'grid_pdnet',
    train_fun,
    eval_fun,
    base_params,
    run_ids=run_ids,
    n_gpus_train=1,
    timeout_train=100,
    n_gpus_eval=1,
    n_samples_eval=100,
    # n_samples=100,
    # n_gpus=1,
)
print(eval_results)
