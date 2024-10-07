# generate dataset
python hamiltonian_nn/slip_wheel_car/generate_singleTrack_6d_dataset.py
# pretrain train ham estimator
python hamiltonian_nn/slip_wheel_car/train_ham_estimator_singletrack6d.py
# pretrain deepreach
python run_experiment.py --mode train --experiment_name SingleTrack6D_pretrain --wandb_project blackbox_deepreach --wandb_entity zeyuanfe --wandb_name SingleTrack6D_pretrain --wandb_group SingleTrack6D --dynamics_class SingleTrack6D --minWith target --pretrain --pretrain_iters 1000 --tMax 1.5  --counter_end 50000 --num_epochs 51000 --num_nl 256 --lr 2e-5 --deepReach_model exact --method NN --set_mode avoid --numpoints 10000 --ham_estimator_fname ham_estimator_singletrack6d_pretrain
# generate augmented dataset
python hamiltonian_nn/slip_wheel_car/generate_singleTrack_6d_dataset.py --USE_VALUE_NET  --value_net_folder_name SingleTrack6D_pretrain
# train ham_estimator again with augmented dataset
python hamiltonian_nn/slip_wheel_car/train_ham_estimator_singletrack6d.py --USE_VALUE_NET  
# # train deepreach
python run_experiment.py --mode train --experiment_name SingleTrack6D_Ham-NN --wandb_project blackbox_deepreach --wandb_entity zeyuanfe --wandb_name SingleTrack6D_Ham-NN --wandb_group SingleTrack6D --dynamics_class SingleTrack6D --minWith target --tMax 1.5  --counter_end 100000 --num_epochs 101000 --num_nl 256 --lr 1e-5 --deepReach_model exact --method NN --set_mode avoid --numpoints 10000 --ham_estimator_fname ham_estimator_singletrack6d --pretrained_model SingleTrack6D_pretrain
# train controller
python hamiltonian_nn/slip_wheel_car/train_control_estimator_singletrack6d.py
# # run verification
python run_experiment.py --mode test --experiment_name SingleTrack6D_Ham-NN --dt 0.0025 --checkpoint_toload -1 --control_type value --data_step run_robust_recovery --wandb_project test --wandb_group test --wandb_name test --wandb_entity zeyuanfe 