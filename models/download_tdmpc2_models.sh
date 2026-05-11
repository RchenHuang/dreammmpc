#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Script directory: $SCRIPT_DIR"

MODEL_DIR="$SCRIPT_DIR/tdmpc2"

mkdir -p "$MODEL_DIR"

tasks=(

    # DMControl tasks
    'acrobot-swingup' 'cartpole-swingup-sparse'  'fish-swim' 'hopper-hop' 'dog-run' 'dog-walk' 'humanoid-run' 'humanoid-walk'
    # Metaworld tasks
    'mw-assembly' 'mw-button-press' 'mw-disassemble' 'mw-lever-pull' 'mw-pick-place-wall' 'mw-push-back' 'mw-shelf-place' 'mw-window-open')

seeds=(1 2 3)


for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        if [[ "$task" == "mw_"* ]]; then
            domain="metaworld"
        else
            domain="dmcontrol"
        fi 

        model_url="https://huggingface.co/nicklashansen/tdmpc2/resolve/main/${domain}/${task}-${seed}.pt?download=true"
        
        local_model_path="$MODEL_DIR/${task}-${seed}.pt"
        
        if [ -f "$local_model_path" ]; then
            echo "Model already exists: $local_model_path"
        else
            wget -O "$local_model_path" "$model_url"
            echo "Downloaded: $local_model_path"
        fi

    done 
done
