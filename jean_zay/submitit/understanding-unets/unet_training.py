from learning_wavelets.training_scripts.unet_training import train_old_unet
from learning_wavelets.evaluation_scripts.unet_evaluate import evaluate_old_unet

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'old_unet_training'
n_epochs = 500
batch_size = 8
base_n_filters = 64
n_layers = 5
layers_n_non_lins = 2
n_gpus = 2
possible_std_dev = [0.0001, 5, 15, 20, 25, 30, 50, 55, 60, 75]

base_parameters = dict(
    n_epochs=n_epochs,
    batch_size=batch_size,
    base_n_filters=base_n_filters,
    n_layers=n_layers,
    layers_n_non_lins=layers_n_non_lins,
)

parameters = [
    base_parameters,
]

res_all = train_eval_grid(
    job_name,
    train_old_unet,
    evaluate_old_unet,
    parameters,
    to_grid=False,
    timeout_train=2,
    n_gpus_train=n_gpus,
    timeout_eval=1,
    n_gpus_eval=1,
    project='learnlets',
    params_to_ignore=['batch_size'],
    noise_std_test=possible_std_dev, 
)

print('Results')
print(res_all)
