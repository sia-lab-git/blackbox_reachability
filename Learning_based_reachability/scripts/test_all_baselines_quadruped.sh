python run_experiment.py --mode test --experiment_name quadruped_1AM1_full --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step run_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_1AM2_full --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step run_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_model_based --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step run_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_1AM3_2 --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step run_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_1AM1_full3 --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step plot_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_1AM2_full2 --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step plot_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_model_based --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step plot_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True

python run_experiment.py --mode test --experiment_name quadruped_1AM3_2 --dt 0.02 --checkpoint_toload -1 --num_scenarios 1000000 --num_violations 10000 --control_type value --data_step plot_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True