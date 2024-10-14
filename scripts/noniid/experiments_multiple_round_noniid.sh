for init_seed in 0 1 2
do
  for partition in noniid-labeldir 
  do
    for beta in 0.5 
    do
        for device in 'cuda:0'
        do
            for epochs in 10 
            do

            python3  noniid_multipleround.py --alpha $beta --model_type mnist_cnn --data mnist --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed --repeat 50
            done
        done
    done
  done
done


##1 day