from learning_wavelets.training_scripts.exact_recon_unet_training import train_unet
from learning_wavelets.evaluation_scripts.exact_recon_unet_evaluate import evaluate_unet

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'exact_recon_unet_training'
n_epochs = 100
base_n_filters = 4
n_layers = 4 
non_linearity = 'relu'
bn = True,
exact_recon = True,
run_id = 'ExactReconUnet_4_bsd500_0_55_None_1620730822'
n_gpus = 1
possible_std_dev = [0.0001, 5, 15, 20, 25, 30, 50, 55, 60, 75]

base_parameters = dict(
    n_epochs=n_epochs,
    batch_size=batch_size,
    base_n_filters=base_n_filters,
    n_layers=n_layers,
    non_linearity=non_linearity,
    bn=bn,
    exact_recon=exact_recon,
)

parameters = [
    base_parameters,
]

res_all = eval_grid(
    job_name,
    evaluate_unet,
    parameters,
    run_ids=[run_id],
    to_grid=False,
    timeout=1,
    n_gpus=n_gpus,
    project='learnlets',
    params_to_ignore=['batch_size'],
    noise_std_test=possible_std_dev, 
)

print('Results')
print(res_all)
