#! /bin/bash
testing_examples=1000
min_index=0
step_size=10
max_index=$((testing_examples - step_size))
t_stim=1000
result_dir=results/IF_curr_exp_t_stim_1000_

mkdir -p $result_dir
#internal parallel loop
for i in $(seq $min_index $step_size $max_index)
do
    python isotope_testing_static.py isotope_model_dense_normalised_input_production_IF_curr_exp_serialised --start_index $i --testing_examples $testing_examples --t_stim $t_stim --time_scale_factor 100 --rate_scaling 1000 --result_dir $result_dir&
done

FILES=$result_dir/*
Accuracy=0
count=0
for file in $FILES
do
    temp_accuracy=$(python ../../OutputDataProcessor.py $file)
    echo $temp_accuracy
    count=$(($count + 1))
done

echo $count
