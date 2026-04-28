#!/bin/bash

# This script should be run via 'ros2 run rl_pipeline demo.sh' after the workspace has been sourced.
seed=12
num_episodes=10
while [ $# -ge 1 ]; do
    case $1 in
        '-s'|'-S'|'--seed')
            shift
            seed="${1}"
            ;;
        '-n'|'-N'|'--num-episodes')
            shift
            num_episodes="${1}"
            ;;
        '-h'|'H'|'--help')
            echo "demo.sh contains the following optional inputs:"
            echo -e "\t-s|-S|--seed\n\t\tSpecifies the seed for the randomly selecting episodes." | fmt
            echo -e "\t-n|-N|--num-episodes\n\t\tSpecifies the number of episodes to run during evaluation." | fmt
            exit
            ;;
        *)
            echo "ERROR: Invalid input argument. Seek help via 'demo.sh --help'"
            exit
            ;;
    esac
    shift;
done
[ $# -gt 0 ] && seed=$1     # allow seed to be passed in as an input argument
[ $# -gt 1 ] && num_episodes=$2
resultfile=/root/ws/src/rbe_capstone/rl_pipeline/demo_results.json
[ -f ${resultfile} ] && rm ${resultfile}

# Define a cleanup function to (aggressively) kill all background processes in case this script is interrupted:
cleanup(){
    [ -f ${resultfile} ] && rm ${resultfile}
    kill -9 -$$ >/dev/null 2>/dev/null
    kill -9 $(pgrep ros2) >/dev/null 2>/dev/null \
        && kill -9 $(pgrep rviz2) >/dev/null 2>/dev/null \
        && pkill -f /opt/ros/jazzy >/dev/null 2>/dev/null \
        && pkill -f python3 >/dev/null 2>/dev/null \
        && kill -9 $$
}
trap cleanup SIGINT SIGTERM EXIT

# Launch evaluation as a background task:
ros2 launch rl_pipeline eval.launch.py \
    use_rviz:=true \
    model:=/root/ws/src/rbe_capstone/rl_pipeline/rl_checkpoints/ppo_nav2_dwb_10000_steps.zip \
    episodes:=${num_episodes} \
    eval_seed:=${seed} \
    results:=${resultfile} \
    >/dev/null 2>/dev/null &    # silents all terminal output and runs the launch file as a background process

# Wait for results to complete and bring them up in gvim:
launch_done=0
baseline_check=""
while [ ${launch_done} -lt 1 ]; do
    [ -f ${resultfile} ] && baseline_check=$(cat ${resultfile} | grep -ho "\"label\": \"Baseline")
    [[ ! -z ${baseline_check} ]] && launch_done=1
done
gvim -f ${resultfile}   # the '-f' should block logic here until the gvim window closes

# After gvim is closed, perform cleanup:
cleanup
