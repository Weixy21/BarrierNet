#!/bin/bash

DATA_ROOT=$HOME/data/vista
TRACE_ROOT=$DATA_ROOT/traces

args=(
    # env
    --trace-paths $TRACE_ROOT/20210527-131252_lexus_devens_center_outerloop
                  $TRACE_ROOT/20210527-131709_lexus_devens_center_outerloop_reverse
                  $TRACE_ROOT/20210609-122400_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-123703_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-133320_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-154745_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-155238_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-175037_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-175503_lexus_devens_outerloop
                  $TRACE_ROOT/20210613-171636_lexus_devens_outerloop
                  $TRACE_ROOT/20210613-172102_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210613-193528_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210726-131322_lexus_devens_center
                  $TRACE_ROOT/20210726-131912_lexus_devens_center_reverse
                  $TRACE_ROOT/20210726-154641_lexus_devens_center
                  $TRACE_ROOT/20210726-155941_lexus_devens_center_reverse
                  $TRACE_ROOT/20210726-184624_lexus_devens_center
                  $TRACE_ROOT/20210726-184956_lexus_devens_center_reverse
                  $TRACE_ROOT/20211114-145502_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-130450_lexus_devens_outerloop_outdoor_wb
                  $TRACE_ROOT/20220113-130914_lexus_devens_outerloop_reverse_outdoor_wb
                  $TRACE_ROOT/20220113-131457_lexus_devens_outerloop
                  $TRACE_ROOT/20220113-131922_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-134438_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-135302_lexus_devens_outerloop
    --mesh-dir $DATA_ROOT/carpack01/
    --n-agents 2
    --reset-mode uniform # default
    --use-curvilinear-dynamics
    --n-episodes 100
    --max-step 100
    --init-dist-range 15 25
    --init-lat-noise-range 1 1.5
    # logging
    --out-dir ./tmp/test
    --use-display
    --save-video
    # barrier net
    --model-module barrier_net
    --ckpt ~/mnt/robosim/results/bnet/barrier_net_tune_5/version_5/checkpoints/epoch\=9-step\=29239.ckpt
    --set-obs-d-lower-bound 1
    # state net
    --state-net-model-module state_net
    --state-net-ckpt ~/mnt/robosim/results/bnet/state_net_9/version_1/checkpoints/epoch=7-step=36374.ckpt
)
args+=("$@")

python eval.py "${args[@]}"