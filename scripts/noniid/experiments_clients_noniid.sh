for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.01 0.05 0.1 0.3 0.5 1
    do
        for device in 'cuda:0'
        do
            for epochs in 200
            do
            for n_parties in  20 50
            do
            python3  noniid.py --alpha $beta --model_type fmnist_cnn --data fmnist --n_nets $n_parties --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            done
            done
        done
    done
  done
done

for init_seed in 0 1 2
do
  for partition in  noniid-#label1  noniid-#label2 noniid-#label3
  do
    for beta in 0.1
    do
        for device in 'cuda:0'
        do
            for epochs in 200
            do
            for n_parties in 20 50
            do
            python3  noniid.py --alpha $beta --model_type fmnist_cnn --data fmnist --n_nets $n_parties --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            done
            done
        done
    done
  done
done

###about 7 days in all