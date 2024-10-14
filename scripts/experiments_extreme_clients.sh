for init_seed in 0 1 2
do
  for partition in noniid-labeldir 
  do
    for beta in  0.001 0.01 0.05 0.1 0.3 0.5 1
    do
      for dataset in fmnist cifar10 mnist svhn
      do
        for device in 'cuda:0'
        do
            for epochs in 200
            do
            for n_parties in  5
            do
            python3 -W ignore experiments_our.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=our \
              --lr=0.01 \
              --batch-size=64 \
              --epochs=$epochs \
              --n_parties=$n_parties \
              --rho=0.9 \
              --comm_round=1 \
              --partition=$partition \
              --beta=$beta \
              --device=$device \
              --datadir='./data/' \
              --logdir='./logs3/' \
              --noise=0 \
              --init_seed=$init_seed \
              --coor=0.999
	            done
          done
        done
      done
    done
  done
done


for init_seed in 0 1 2
do
  for partition in  noniid-#label3
  do
    for beta in 0.1
    do
      for dataset in fmnist cifar10 mnist svhn
      do
        for device in 'cuda:0'
        do
            for epochs in 200
            do
            for n_parties in 5
            do
            python3 -W ignore experiments_our.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=blockofflinenewton \
              --lr=0.01 \
              --batch-size=64 \
              --epochs=$epochs \
              --n_parties=$n_parties \
              --rho=0.9 \
              --comm_round=1 \
              --partition=$partition \
              --beta=$beta \
              --device=$device \
              --datadir='./data/' \
              --logdir='./logs3/' \
              --noise=0 \
              --init_seed=$init_seed \
              --coor=0.999
	            done
          done
        done
      done
    done
  done
done



for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.001 0.01 0.05 0.1 0.3 0.5 1.0
    do
      for epochs in  200
      do
        for device in 'cuda:0'
        do
    	  for  dataset in fmnist cifar10 mnist svhn
          do
          for n_parties in 5
          do
            for alg in fednova scaffold fedavg
            do
              python3 experiments.py --model=simple-cnn \
                --dataset=$dataset \
                --alg=$alg \
                --lr=0.01 \
                --batch-size=64 \
                --epochs=$epochs  \
                --n_parties=10 \
                --rho=0.9 \
                --comm_round=1 \
                --partition=$partition \
                --beta=$beta \
                --device=$device \
                --datadir='./data/' \
                --logdir='./logs/' \
                --noise=0 \
                --init_seed=$init_seed
            done
          done
          done
        done
      done
    done
  done
done

for init_seed in 0 1 2
do
  for partition in  noniid-#label3
  do
    for beta in 0.1
    do
      for epochs in  200
      do
        for device in 'cuda:0'
        do
    	  for  dataset in fmnist cifar10 mnist svhn
          do
          for n_parties in  5
          do
            for alg in fednova scaffold fedavg
            do
              python3 experiments.py --model=simple-cnn \
                --dataset=$dataset \
                --alg=$alg \
                --lr=0.01 \
                --batch-size=64 \
                --epochs=$epochs  \
                --n_parties=10 \
                --rho=0.9 \
                --comm_round=1 \
                --partition=$partition \
                --beta=$beta \
                --device=$device \
                --datadir='./data/' \
                --logdir='./logs/' \
                --noise=0 \
                --init_seed=$init_seed
            done
          done
          done
        done
      done
    done
  done
done



for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.001  0.01 0.05 0.1 0.3 0.5 1.0
    do
      for epochs in 200
      do
        for device in 'cuda:0'
        do
          for dataset in fmnist cifar10 mnist svhn
          do
          for n_parties in 5
          do
            for mu in 0.001 0.01 0.1 1
            do
              python3 experiments.py --model=simple-cnn \
                --dataset=$dataset \
                --alg=fedprox \
                --lr=0.01 \
                --batch-size=64 \
                --epochs=$epochs  \
                --n_parties=10 \
                --rho=0.9 \
                --mu=$mu \
                --comm_round=1 \
                --partition=$partition \
                --beta=$beta \
                --device=$device \
                --datadir='./data/' \
                --logdir='./logs/' \
                --noise=0 \
                --init_seed=$init_seed
            done
          done
          done
        done
      done
    done
  done
done

for init_seed in 0 1 2
do
  for partition in  noniid-#label3
  do
    for beta in  0.1
    do
      for epochs in 200
      do
        for device in 'cuda:0'
        do
          for dataset in fmnist cifar10 mnist svhn
          do
          for n_parties in 5
          do
            for mu in 0.001 0.01 0.1 1
            do
              python3 experiments.py --model=simple-cnn \
                --dataset=$dataset \
                --alg=fedprox \
                --lr=0.01 \
                --batch-size=64 \
                --epochs=$epochs  \
                --n_parties=10 \
                --rho=0.9 \
                --mu=$mu \
                --comm_round=1 \
                --partition=$partition \
                --beta=$beta \
                --device=$device \
                --datadir='./data/' \
                --logdir='./logs/' \
                --noise=0 \
                --init_seed=$init_seed
            done
          done
          done
        done
      done
    done
  done
done


#If you want to run parallel, plz give each script a unique save_dir
for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.001 0.01 0.05 0.1 0.3 0.5 1.0
    do
        for device in 'cuda:0'
        do
            for epochs in  200
            do
            for n_parties in 5
            do

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=fmnist_cnn  --dataset=fmnist --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/fmnist  --other=fmnist --model=fmnist_cnn --dataset=fmnist --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=mnist_cnn  --dataset=mnist --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/mnist  --other=mnist --model=mnist_cnn --dataset=mnist --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=svhn_cnn  --dataset=svhn --beta=$beta  --seed=$init_seed --num_users=$n_parties --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/svhn --other=svhn --model=svhn_cnn --dataset=svhn --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            done
            done
        done
    done
  done
done

for init_seed in 0 1 2
do
  for partition in  noniid-#label3
  do
    for beta in  0.1
    do
        for device in 'cuda:0'
        do
            for epochs in 5
            do
            for n_parties in 20 50
            do
            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=fmnist_cnn  --dataset=fmnist --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/fmnist  --other=fmnist --model=fmnist_cnn --dataset=fmnist --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=mnist_cnn  --dataset=mnist --beta=$beta  --seed=$init_seed --num_users=$n_parties  --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/mnist  --other=mnist --model=mnist_cnn --dataset=mnist --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=svhn_cnn  --dataset=svhn --beta=$beta  --seed=$init_seed --num_users=$n_parties --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/svhn --other=svhn --model=svhn_cnn --dataset=svhn --adv=1 --beta=$beta --seed=$init_seed --num_users  $n_parties   --partition  $partition

            done
            done
        done
    done
  done
done


###about 4 days in all