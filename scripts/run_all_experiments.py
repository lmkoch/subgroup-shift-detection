import pathlib
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run all experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_base_dir", action="store", type=str, help="experiment folder", default='./experiments'
    )      
    parser.add_argument(
        "--cmd_file", action="store", type=str, help="executed commands are logged here", default='cmds.log'
    )    
    parser.add_argument(
        "--slurm", action="store_true", dest='slurm', help="slurm flag", default=False
    )
    parser.add_argument(
        "--slurm_email", action="store", type=str, help="Needed for slurm only"
    )
    parser.add_argument(
        "--dry_run", action="store", type=bool, help="if flag is set, nothing is actually executed", default=False
    )    
    parser.add_argument('--odin', dest='odin', action='store_true', 
                        help="ODIN flag", default=True)    
    parser.add_argument('--no-odin', dest='odin', action='store_false', 
                        help="ODIN flag", default=True)
    parser.add_argument('--mmd_hyperparam', dest='mmd_hyperparam', action='store_true', 
                        help="MMD-D hyperparameter flag", default=True)    
    parser.add_argument('--no-mmd_hyperparam', dest='mmd_hyperparam', action='store_false', 
                        help="MMD-D hyperparameter flag", default=True)
    parser.add_argument('--hypo_mnist', dest='hypo_mnist', action='store_true', 
                        help="hypothesis test experiments on MNIST flag", default=True)    
    parser.add_argument('--no-hypo_mnist', dest='hypo_mnist', action='store_false', 
                        help="hypothesis test experiments on MNIST flag", default=True)
    parser.add_argument('--hypo_camelyon', dest='hypo_camelyon', action='store_true', 
                        help="hypothesis test experiments on Camelyon17 flag", default=True)    
    parser.add_argument('--no-hypo_camelyon', dest='hypo_camelyon', action='store_false', 
                        help="hypothesis test experiments on Camelyon17 flag", default=True)
                   
    args = parser.parse_args()
           
    
    cmds_prep = []
    
    exp_base_dir = args.exp_base_dir
    
    # prepare experiment configurations
    cmds_prep.append(f"python ./scripts/prepare_main_exps.py --exp_base_dir {exp_base_dir}")
    
    #######################################################################################
    # Experiment 1: individual OOD detection
    #######################################################################################
    
    cmds = []

    if args.odin:
        exp_dir = os.path.join(exp_base_dir, 'individual-ood')
        cmds.append(f"python ./scripts/run_odin_exp.py --exp_dir {exp_dir} \
            --config_file ./config/odin_mnist_5x100.yaml")
        cmds.append(f"python ./scripts/run_odin_exp.py --exp_dir {exp_dir} \
            --config_file ./config/odin_mnist_no5.yaml")
        
    #######################################################################################
    # Experiment 2: population-level subgroup shift detection
    #######################################################################################

    # run them all
    
    hyperparam_dir = os.path.join(exp_base_dir, 'hypothesis-tests/mnist_hyperparam')
    mnist_dir = os.path.join(exp_base_dir, 'hypothesis-tests/mnist')
    camelyon_dir = os.path.join(exp_base_dir, 'hypothesis-tests/camelyon')

    if args.mmd_hyperparam:
        # hyperparameter sweep on validation set
        for config_file in pathlib.Path(hyperparam_dir).glob('**/config.yaml'):
            cmds.append(f"python ./scripts/run_main_exp.py --exp_dir {hyperparam_dir} \
                --config_file {config_file} \
                --eval_splits validation")

    # final experiments on test set
    if args.hypo_mnist:
        for config_file in pathlib.Path(mnist_dir).glob('**/config.yaml'):
            cmds.append(f"python ./scripts/run_main_exp.py --exp_dir {mnist_dir} \
                --config_file {config_file} \
                --eval_splits test") 

    if args.hypo_camelyon:
        for config_file in pathlib.Path(camelyon_dir).glob('**/config.yaml'):
            cmds.append(f"python ./scripts/run_main_exp.py --exp_dir {camelyon_dir} \
                --config_file {config_file} \
                --eval_splits test") 
                   
    if args.slurm:
        # slurm runner wrapper around compute intensive commands
        cmds = [f'python ./scripts/slurm_runner.py --email {args.slurm_email} -- {ele}' for ele in cmds]
        
    with open(args.cmd_file, 'w') as f:
        all_commands = '\n'.join(cmds_prep + cmds)
        print(all_commands)
        f.write(all_commands)
    
    if not args.dry_run:
        os.system(" && ".join(cmds_prep + cmds))
    