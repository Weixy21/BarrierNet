#!/bin/bash

# Example usage: bash run.sh mat 2 0

# Parse arguments
MODE=$1
NUM=$2 # end index of traces to be collected
START_NUM=${3:-0}

DSET_ROOT=/home/tsunw/data/bnet_dataset_full
MAT_DIR=$DSET_ROOT/mat
FROZEN_SIM_DIR=$DSET_ROOT/frozen_sim
CONTROL_DIR=$DSET_ROOT/control
IMGS_DIR=$DSET_ROOT/images

TRACE_ROOT=$HOME/data/vista/traces
ALL_TRACES=( # 25 traces in total
    "$TRACE_ROOT/20210527-131252_lexus_devens_center_outerloop"
    "$TRACE_ROOT/20210527-131709_lexus_devens_center_outerloop_reverse"
    "$TRACE_ROOT/20210609-122400_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20210609-123703_lexus_devens_outerloop"
    "$TRACE_ROOT/20210609-133320_lexus_devens_outerloop"
    "$TRACE_ROOT/20210609-154745_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20210609-155238_lexus_devens_outerloop"
    "$TRACE_ROOT/20210609-175037_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20210609-175503_lexus_devens_outerloop"
    "$TRACE_ROOT/20210613-171636_lexus_devens_outerloop"
    "$TRACE_ROOT/20210613-172102_lexus_devens_outerloop_reverse"
    # "$TRACE_ROOT/20210613-193120_lexus_devens_outerloop" # stuck at randomly_place_agent; no complete trace
    "$TRACE_ROOT/20210613-193528_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20210726-131322_lexus_devens_center"
    "$TRACE_ROOT/20210726-131912_lexus_devens_center_reverse"
    "$TRACE_ROOT/20210726-154641_lexus_devens_center"
    "$TRACE_ROOT/20210726-155941_lexus_devens_center_reverse"
    "$TRACE_ROOT/20210726-184624_lexus_devens_center"
    "$TRACE_ROOT/20210726-184956_lexus_devens_center_reverse"
    "$TRACE_ROOT/20211114-145502_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20220113-130450_lexus_devens_outerloop_outdoor_wb"
    "$TRACE_ROOT/20220113-130914_lexus_devens_outerloop_reverse_outdoor_wb"
    "$TRACE_ROOT/20220113-131457_lexus_devens_outerloop"
    "$TRACE_ROOT/20220113-131922_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20220113-134438_lexus_devens_outerloop_reverse"
    "$TRACE_ROOT/20220113-135302_lexus_devens_outerloop"
)
N_PER_TRACE=5

if [ "$MODE" = "mat" ]; then
    echo "Collect mat"
    for i in $(seq -f "%03g" $((START_NUM+0)) 1 $((NUM-1)))
    do
        INT_I=10#$i
        TRACE_I=$((INT_I/N_PER_TRACE))
        TRACE_PATH="${ALL_TRACES[TRACE_I]}"
        SCENE_IDX=$(printf "%02d" $TRACE_I)
        TRACE_IDX=$(printf "%02d" $((INT_I%N_PER_TRACE)))
        args=(
            --trace-paths $TRACE_PATH
            --mesh-dir "$HOME/data/vista/carpack01/"
            --n-agents 40
            --mode "collect_init"
            --init-mat-path "$MAT_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}.mat"
            --init-frozen-sim-path "$FROZEN_SIM_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_frozen_sim.pkl"
            --visualize-privileged-info
        )
        python main.py "${args[@]}"
    done
elif [ "$MODE" = "mat_lk" ]; then
    echo "Collect mat (lane keeping)"
    for i in $(seq -f "%03g" $((START_NUM+0)) 1 $((NUM-1)))
    do
        N_PER_TRACE=1
        INT_I=10#$i
        TRACE_I=$((INT_I/N_PER_TRACE))
        TRACE_PATH="${ALL_TRACES[TRACE_I]}"
        SCENE_IDX=$(printf "%02d" $TRACE_I)
        TRACE_IDX="-1"
        args=(
            --trace-paths $TRACE_PATH
            --mesh-dir "$HOME/data/vista/carpack01/"
            --n-agents 1
            --mode "collect_init"
            --init-mat-path "$MAT_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}.mat"
            --init-frozen-sim-path "$FROZEN_SIM_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_frozen_sim.pkl"
            --visualize-privileged-info
        )
        python main.py "${args[@]}"
    done
elif [ "$MODE" = "imgs" ]; then
    echo "Collect images"
    for i in $(seq -f "%03g" $((START_NUM+0)) 1 $((NUM-1)))
    do
        echo "$i"
        INT_I=10#$i
        TRACE_I=$((INT_I/N_PER_TRACE))
        TRACE_PATH="${ALL_TRACES[TRACE_I]}"
        SCENE_IDX=$(printf "%02d" $TRACE_I)
        TRACE_IDX=$(printf "%02d" $((INT_I%N_PER_TRACE)))
        SEQ_START_NUM=0
        SEQ_NUM=39
        for j in $(seq -f "%02g" $((SEQ_START_NUM+0)) 1 $((SEQ_NUM-1)))
        do
            args=(
                --trace-paths $TRACE_PATH
                --mesh-dir "$HOME/data/vista/carpack01/"
                --n-agents 40
                --mode "collect_imgs"
                --load-frozen-sim "$FROZEN_SIM_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_frozen_sim.pkl"
                --load-control "$CONTROL_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_seq_${j}.mat"
                --imgs-dir "$IMGS_DIR/scene_${SCENE_IDX}/trace_${TRACE_IDX}/seq_${j}"
            )
            python main.py "${args[@]}"
        done
    done
elif [ "$MODE" = "imgs_lk" ]; then
    echo "Collect images (lane keeping)"
    for i in $(seq -f "%03g" $((START_NUM+0)) 1 $((NUM-1)))
    do
        echo "$i"
        INT_I=10#$i
        TRACE_I=$((INT_I/N_PER_TRACE))
        TRACE_PATH="${ALL_TRACES[TRACE_I]}"
        SCENE_IDX=$(printf "%02d" $TRACE_I)
        TRACE_IDX="-1"
        SEQ_START_NUM=0
        SEQ_NUM=100
        for j in $(seq -f "%02g" $((SEQ_START_NUM+0)) 1 $((SEQ_NUM-1)))
        do
            args=(
                --trace-paths $TRACE_PATH
                --mesh-dir "$HOME/data/vista/carpack01/"
                --n-agents 40
                --mode "collect_imgs"
                --load-frozen-sim "$FROZEN_SIM_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_frozen_sim.pkl"
                --load-control "$CONTROL_DIR/scene_${SCENE_IDX}_trace_${TRACE_IDX}_seq_${j}.mat"
                --imgs-dir "$IMGS_DIR/scene_${SCENE_IDX}/trace_${TRACE_IDX}/seq_${j}"
            )
            python main.py "${args[@]}"
        done
    done
else
    echo "Not implemented"
fi
