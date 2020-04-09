#! /bin/bash
testing_examples=1692
min_index=0
step_size=18
max_index=$((testing_examples - step_size + min_index))
t_stim=200
result_dir=results/IF_curr_exp_t_stim_$(t_stim)_testing_examples_$(testing_examples)

mkdir -p $result_dir
#internal parallel loop
for i in $(seq $min_index $step_size $max_index)
do
    echo $i
    python isotope_testing_static.py isotope_model_dense_normalised_input_production_IF_curr_exp_serialised --start_index $i --testing_examples $step_size --t_stim $t_stim --time_scale_factor 100 --rate_scaling 1000 --result_dir $result_dir&
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
