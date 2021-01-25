from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet
from fastmri_recon.evaluate.scripts.xpdnet_inference import xpdnet_inference
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid, infer_grid


job_name = 'xpdnet_flmd'
loss = 'compound_mssim'
lr = 1e-4
batch_size = None
n_samples = None
n_epochs = 300
n_primal = 5
contrast = None
refine_smaps = True
refine_big = False
n_dual = 3
primal_only = True
multiscale_kspace_learning = False
n_dual_filters = 8
n_iter_per_model_size = {
    'small': 24,
    'medium': 19,
    'big': 7,
    'XL': 2,
}
use_mixed_precision = False
model_specs = list(get_model_specs(n_primal=n_primal))

parameter_grid = [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_samples=n_samples,
        refine_smaps=refine_smaps,
        refine_big=refine_big,
        af=8,
        loss=loss,
        lr=lr,
        mask_type='random',
        n_dual=n_dual,
        n_iter=n_iter_per_model_size[model_size],
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
        multiscale_kspace_learning=multiscale_kspace_learning,
        use_mixed_precision=use_mixed_precision,
    ) for model_name, model_size, model_fun, kwargs, _, n_scales, res in model_specs
    if model_size != 'XL' and 'MWCNN' not in model_name
] + [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_samples=n_samples,
        refine_smaps=refine_smaps,
        refine_big=refine_big,
        af=4,
        loss=loss,
        lr=lr,
        mask_type='random',
        n_dual=n_dual,
        n_iter=n_iter_per_model_size[model_size],
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
        multiscale_kspace_learning=multiscale_kspace_learning,
        use_mixed_precision=use_mixed_precision,
    ) for model_name, model_size, model_fun, kwargs, _, n_scales, res in model_specs
    if model_size != 'XL' and 'MWCNN' not in model_name
]

run_ids = [
    'xpdnet_singlecoil__af8_i7_compound_mssim_rf_sm_UnetMultiDomainbig_1610912245',
    'xpdnet_singlecoil__af8_i19_compound_mssim_rf_sm_UnetMultiDomainmedium_1610912245',
    'xpdnet_singlecoil__af8_i24_compound_mssim_rf_sm_UnetMultiDomainsmall_1610922641',
    'xpdnet_singlecoil__af4_i7_compound_mssim_rf_sm_UnetMultiDomainbig_1610912352',
    'xpdnet_singlecoil__af4_i19_compound_mssim_rf_sm_UnetMultiDomainmedium_1610931952',
    'xpdnet_singlecoil__af4_i24_compound_mssim_rf_sm_UnetMultiDomainsmall_1610931952',
]
eval_results, run_ids = train_eval_grid(
# eval_results = eval_grid(
    job_name,
    train_xpdnet,
    evaluate_xpdnet,
    parameter_grid,
    # run_ids=run_ids,
    n_samples_eval=50,
    timeout_train=70,
    n_gpus_train=batch_size if batch_size else 1,
    timeout_eval=10,
    n_gpus_eval=1,
    # n_samples=50,
    # timeout=10,
    # n_gpus=1,
    to_grid=False,
    return_run_ids=True,
    checkpoints_train=2,
    resume_checkpoint=2,
    resume_run_run_ids=run_ids,
    params_to_ignore=['batch_size', 'use_mixed_precision'],
)

print(eval_results)

# infer_grid(
#     job_name,
#     xpdnet_inference,
#     parameter_grid,
#     run_ids=run_ids,
#     timeout=10,
#     n_gpus=4,
#     to_grid=False,
#     params_to_ignore=['mask_type', 'multicoil', 'batch_size', 'use_mixed_precision'],
# )
