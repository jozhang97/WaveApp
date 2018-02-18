#TODO Rename this 

#!/usr/bin/env sh
torch_home=./data
model=alexnet
dataset=vctk
epochs=300

$PYTHON ./main.py {torch_home}/vctk \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} --learning_rate 0.05 \
	--schedule 150 225 --gammas 0.1 0.1 --batch_size 64 --workers 16 --ngpu 1
