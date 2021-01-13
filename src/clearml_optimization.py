from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from utils import cfg
from clearml import Task


def optimize():
    task = Task.init(project_name='ariel-mde',
                     task_name='Hyperparameter Search',
                     task_type=Task.TaskTypes.optimizer,
                     reuse_last_task_id=True)
    # experiment template to optimize in the hyper-parameter optimization
    args = {
        'template_task_id': '9b56f8b81c254323a4e0b7e52ca45734',
        'run_as_service': False,
    }
    args = task.connect(args)

    # Get the template task experiment that we want to optimize
    if not args['template_task_id']:
        args['template_task_id'] = Task.get_task(project_name='ariel-mde', task_name='').id
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=args['template_task_id'],
        hyper_parameters=[
            UniformIntegerParameterRange('train/batch_size', min_value=2, max_value=10, step_size=2),
            DiscreteParameterRange('model/use_bn', [True, False]),
            DiscreteParameterRange('model/use_double_bn', [True, False]),
            UniformParameterRange('model/dropout', min_value=0, max_value=0.5, step_size=0.05),
            UniformParameterRange('optim/lr', min_value=1e-5, max_value=1e-1),
            UniformIntegerParameterRange('misc/random_seed', min_value=1, max_value=96, step_size=5),
            UniformParameterRange('data_augmentation/flip_p', min_value=0, max_value=0.5, step_size=0.1)
            # UniformParameterRange('config/data_augmentation/horizontal_flip', min_value=0, max_value=1),
            # UniformParameterRange('config/data_augmentation/color_jitter', min_value=0, max_value=1),
            # UniformParameterRange('config/data_augmentation/gaussian_blur', min_value=0, max_value=1),
            # UniformParameterRange('config/data_augmentation/gaussian_noise', min_value=0, max_value=1),
        ],
        objective_metric_title=cfg['optim']['loss'],
        objective_metric_series='loss',
        objective_metric_sign='min',
        max_number_of_concurrent_tasks=2,
        optimizer_class=OptimizerOptuna,
        # Optional: Limit the execution time of a single experiment, in minutes.
        # (this is optional, and if using  OptimizerBOHB, it is ignored)
        time_limit_per_job=100.,
        # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=3,
        # set the maximum number of jobs to launch for the optimization, default (None) unlimited
        # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
        # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
        total_max_jobs=30,
        min_iteration_per_job=10,  # minimum number of iterations per experiment, till early stopping
        max_iteration_per_job=1000,  # maximum number of iterations per experiment
    )
    optimizer.set_report_period(1)  # setting the time gap between two consecutive reports
    optimizer.start()
    optimizer.wait()  # wait until process is done
    optimizer.stop()  # make sure background optimization stopped
    # optimization is completed, print the top performing experiments id
    k = 3
    top_exp = optimizer.get_top_experiments(top_k=k)
    print('Top {} experiments are:'.format(k))
    for n, t in enumerate(top_exp, 1):
        print('Rank {}: task id={} |result={}'
              .format(n, t.id, t.get_last_scalar_metrics()[cfg['optim']['loss']]['total']['last']))


if __name__ == '__main__':
    optimize()
