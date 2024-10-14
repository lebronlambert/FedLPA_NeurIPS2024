for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.001
    do
        for device in 'cuda:0'
        do
            for epochs in 10 20 50 100 200
            do
            python3  noniid.py --alpha $beta --model_type fmnist_cnn --data fmnist --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            python3  noniid.py --alpha $beta --model_type cifar10_cnn --data cifar10 --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            python3  noniid.py --alpha $beta --model_type mnist_cnn --data mnist --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            python3  noniid.py --alpha $beta --model_type svhn_cnn --data svhn --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=$epochs  --partition $partition   --device $device --seed $init_seed
            done
        done
    done
  done
done


##about 2days