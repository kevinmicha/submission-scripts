from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_large'
n_gpus = 4
n_runs = 5
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=220,
    save_at=50,
)
for i_run in range(n_runs):
    parameters += [
        dict(seed=i_run, **base_params),
        dict(seed=i_run, shine=True, **base_params),
        dict(seed=i_run, fpn=True, **base_params),
    ]

train_eval_grid(
    job_name,
    train_classifier,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout_train=20,
    n_gpus_train=n_gpus,
    timeout_eval=2,
    n_gpus_eval=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs', 'save_at'],
    torch=True,
)
