#!/usr/bin/env python3

"""
Simpler wrapper scripts which automatically sets up the jobs
for running on slurm

To run something on slurm you just call
./slurm_run.py --gpus x -- python main_program_to_run.py --main_args

The script assumes you are using conda and you start the script from
the correct conda environment. This conda environment needs to be accessible
from SLURM and needs to be named the same.

This script expects to find `--basedir` and `--expname` in your 
`main_program_to_run.py` arguments. The slurm logs are then created in basedir/expname.
If not provided, defaults to ./slurmlog/timestamp.

A simple parser which ensure this is available is defined in `base_parser()`. You can
remove this and just include your own parser and use it in `get_jobname_and_logdir()`.


This script was adapted from the original script provided by 
Mark Boss (mark.boss@uni-tuebingen.de)

"""

import argparse
import os
import sys
from typing import Tuple
import time


def base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--basedir",
        default='./slurmlog',
        type=str,
        help="Basedir to store the slurm logs. Runs are organized in folder EXPNAME.",
    )
    parser.add_argument(
        "--expname", default=f'{time.time()}', type=str, help="log folder to create."
    )

    return parser


def get_jobname_and_logdir(args):
    bp = base_parser()
    train_args = bp.parse_known_args(args)[0]

    expname = train_args.expname
    basedir = train_args.basedir

    logdir = os.path.join(basedir, expname)

    return expname, logdir


def split_runner_main_args():
    cur_args = sys.argv

    # Find the arguments belonging to the runner
    # and main args
    split_idx = cur_args.index("--")
    # Split
    # Ignore script name
    runner_args = cur_args[1:split_idx]
    # Do not add the split argument
    main_args = cur_args[split_idx + 1 :]

    return runner_args, main_args


def main_args_to_string(args):
    return " ".join(args)


def parse_args():
    runner_args, main_args = split_runner_main_args()

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument(
        "--partition",
        default="gpu-2080ti",
        choices=[
            "gpu-2080ti",
            "gpu-2080ti-long",
            "gpu-2080ti-dev",
            "gpu-2080ti-preemptable",
            "gpu-v100",
        ],
    )

    parser.add_argument(
        "--duration_hours",
        type=int,
        default=72,
        help="Maximum run-time in hours. Run will be killed after that amount of time. "
        "Make sure not to exceed the maximum-allowed time of the selected partition!",
    )

    parser.add_argument(
        "--email",
        type=str,
        help=(
            "Email to which the slurm-cluster sends status updates (END). "
            "By default the script looks for the environment variable SLURM_MAIL"
        ),
        default=os.environ.get("SLURM_MAIL"),
    )

    parser.add_argument(
        "--base_mem",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--base_cpu",
        type=int,
        default=4,
    )

    return parser.parse_args(runner_args), main_args


def get_current_conda_environment():
    return os.environ["CONDA_DEFAULT_ENV"]


def compute_node_configuration(
    gpus: int, partition: str, base_mem_gb=32, base_cpu=4
) -> Tuple[int, int]:
    # all partition only have 8-gpu machines

    mem = min(int(gpus * base_mem_gb), 128)
    cpus_per_task = base_cpu * gpus
    if partition == "gpu-v100":
        cpus_per_task = int(min(64, cpus_per_task))
    else:
        cpus_per_task = int(min(72, cpus_per_task))
    return mem, cpus_per_task


def slugify(str):
    return "".join(x for x in str if x.isalnum())


def get_out_err_job_file(expname, logdir):
    out_file = os.path.join(logdir, slugify(expname) + "-%j.out")
    err_file = os.path.join(logdir, slugify(expname) + "-%j.err")
    job_file = os.path.join(logdir, slugify(expname) + ".sh")

    return out_file, err_file, job_file


def write_sbatch_property(sbatch_property, value, sbatch_file):
    """Writes the value of a sbatch property into a file."""
    sbatch_file.writelines("#SBATCH --{}={}\n".format(sbatch_property, value))


def build_submit_command(runner_args, main_args):
    expname, logdir = get_jobname_and_logdir(main_args)
    # Ensure log dir exists
    os.makedirs(logdir, exist_ok=True)

    out_file, err_file, job_file = get_out_err_job_file(expname, logdir)

    mem, cpus_per_run = compute_node_configuration(
        runner_args.gpus,
        runner_args.partition,
        runner_args.base_mem,
        runner_args.base_cpu,
    )

    time = "{}-{}".format(
        runner_args.duration_hours // 24, runner_args.duration_hours % 24
    )

    with open(job_file, "w") as sbatch_file:
        sbatch_file.writelines("#!/bin/bash\n")
        write_sbatch_property("job-name", slugify(expname), sbatch_file)
        write_sbatch_property("partition", runner_args.partition, sbatch_file)
        write_sbatch_property("ntasks", 1, sbatch_file)
        write_sbatch_property("time", time, sbatch_file)
        # write_sbatch_property("gpus", arguments.gpus_per_experiment, sbatch_file)
        write_sbatch_property("gres", "gpu:{}".format(runner_args.gpus), sbatch_file)
        write_sbatch_property("mem", "{}G".format(mem), sbatch_file)
        write_sbatch_property("cpus-per-task", cpus_per_run, sbatch_file)
        write_sbatch_property("output", out_file, sbatch_file)
        write_sbatch_property("error", err_file, sbatch_file)
        write_sbatch_property("mail-type", "END", sbatch_file)
        write_sbatch_property("mail-user", runner_args.email, sbatch_file)
        # Body
        sbatch_file.writelines("\n")
        sbatch_file.writelines("scontrol show job ${SLURM_JOB_ID}\n\n")
        sbatch_file.writelines(
            [
                "source ~/.bashrc \n",
                "which conda\n" "conda env list\n" "nvidia-smi\n\n",
                # "cd {}\n".format(os.path.dirname(os.path.realpath(__file__))),
                "ls -l\n\n"
                "conda activate {}\n".format(get_current_conda_environment()),
                "which python\n\n",
                " ".join(main_args) + "\n",
            ]
        )

    cmd_args = ["sbatch"]
    cmd_args.append("--verbose")
    cmd_args.append(job_file)

    # Ensure everything is a string
    cmd_args = [str(s) for s in cmd_args]

    cmd = " ".join(cmd_args)

    os.system(cmd)


if __name__ == "__main__":
    args, main_args = parse_args()

    assert args.email is not None

    build_submit_command(args, main_args)
