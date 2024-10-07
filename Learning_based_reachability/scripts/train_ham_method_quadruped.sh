# generate dataset 
python data/data_collection_quadruped/collect_dataset.py
# train autoencoder
python hamiltonian_nn/quadruped/train_autoencoder.py
# above are necessary for all methods
# pretrain train ham estimator
python hamiltonian_nn/quadruped/train_ham_estimator_quadruped.py
# pretrain deepreach
python run_experiment.py --mode train --experiment_name quadruped_pretrain --wandb_project blackbox_deepreach --wandb_entity zeyuanfe --wandb_name quadruped_pretrain --wandb_group Quadruped  --dynamics_class Quadruped --minWith target --tMax 0.6  --pretrain --pretrain_iters 1000 --counter_end 10000 --num_epochs 12000  --num_nl 512 --lr 2e-5 --deepReach_model exact --method NN --set_mode avoid --collisionR 0.5 --val_time_resolution 6 --ham_estimator_fname ham_estimator_pretrained
# train ham_estimator again with augmented dataset
python hamiltonian_nn/quadruped/train_ham_estimator_quadruped.py --USE_VALUE_NET  
# train deepreach
python run_experiment.py --mode train --experiment_name quadruped_Ham-NN --wandb_project blackbox_deepreach --wandb_entity zeyuanfe --wandb_name quadruped_Ham-NN --wandb_group Quadruped  --dynamics_class Quadruped --minWith target --tMax 0.6  --counter_end 80000 --num_epochs 81000 --num_nl 512 --lr 6e-6 --deepReach_model exact --method NN --set_mode avoid --collisionR 0.5 --val_time_resolution 6 --ham_estimator_fname ham_estimator_quadruped --pretrained_model quadruped_pretrain
# train controller
python hamiltonian_nn/quadruped/train_control_estimator_quadruped.py --value_net_folder_name quadruped_Ham-NN
# run verification
python run_experiment.py --mode test --experiment_name quadruped_Ham-NN --dt 0.02 --checkpoint_toload -1 --control_type value --data_step run_robust_recovery --num_scenarios 1000000 --num_violations 10000 --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe --use_ISAAC --headless True