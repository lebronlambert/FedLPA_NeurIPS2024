for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.01 0.05 0.1 0.3 0.5 1
    do
      for dataset in fmnist cifar10 mnist svhn
      do
        for device in 'cuda:0'
        do
            for epochs in 10 20 50 100 200
            do
            python3 -W ignore experiments_our.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=our \
              --lr=0.01 \
              --batch-size=64 \
              --epochs=$epochs \
              --n_parties=10 \
              --rho=0.9 \
              --comm_round=1 \
              --partition=$partition \
              --beta=$beta \
              --device=$device \
              --datadir='./data/' \
              --logdir='./logs/' \
              --noise=0 \
              --init_seed=$init_seed \
              --coor=0.999

          done
        done
      done
    done
  done
done
#about 6 days

for init_seed in 0 1 2
do
  for partition in noniid-#label1  noniid-#label2  noniid-#label3
  do
    for beta in  0.1
    do
      for dataset in fmnist cifar10 mnist svhn
      do
        for device in 'cuda:0'
        do
            for epochs in 10 20 50 100 200
            do
            python3 -W ignore experiments_our.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=our \
              --lr=0.01 \
              --batch-size=64 \
              --epochs=$epochs \
              --n_parties=10 \
              --rho=0.9 \
              --comm_round=1 \
              --partition=$partition \
              --beta=$beta \
              --device=$device \
              --datadir='./data/' \
              --logdir='./logs/' \
              --noise=0 \
              --init_seed=$init_seed \
              --coor=0.999

          done
        done
      done
    done
  done
done
#about 3 days


for init_seed in 0 1 2
do
  for partition in noniid-labeldir 
  do
    for beta in  0.01 0.05 0.1 0.3 0.5 1.0
    do
      for epochs in 10 20 50 100 200
      do
        for device in 'cuda:0'
        do
    	  for  dataset in fmnist cifar10 mnist svhn
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
#about 12days

for init_seed in 0 1 2
do
  for partition in noniid-#label1  noniid-#label2  noniid-#label3
  do
    for beta in 0.1
    do
      for epochs in 10 20 50 100 200
      do
        for device in 'cuda:0'
        do
    	  for  dataset in fmnist cifar10 mnist svhn
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
#about 6days


for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.01 0.05 0.1 0.3 0.5 1.0
    do
      for epochs in 10 20 50 100 200
      do
        for device in 'cuda:0'
        do
          for dataset in fmnist cifar10 mnist svhn
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
#about 12days

for init_seed in 0 1 2
do
  for partition in noniid-#label1  noniid-#label2  noniid-#label3
  do
    for beta in  0.1
    do
      for epochs in 10 20 50 100 200
      do
        for device in 'cuda:0'
        do
          for dataset in fmnist cifar10 mnist svhn
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

#about 6days

#If you want to run parallel, plz give each script a unique save_dir
for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.01 0.05 0.1 0.3 0.5 1.0
    do
        for device in 'cuda:0'
        do
            for epochs in 10 20 50 100 200
            do

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=fmnist_cnn  --dataset=fmnist --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/fmnist  --other=fmnist --model=fmnist_cnn --dataset=fmnist --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=mnist_cnn  --dataset=mnist --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/mnist  --other=mnist --model=mnist_cnn --dataset=mnist --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=svhn_cnn  --dataset=svhn --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/svhn --other=svhn --model=svhn_cnn --dataset=svhn --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition


            done
        done
    done
  done
done
#about 24days

for init_seed in 0 1 2
do
  for partition in  noniid-#label1  noniid-#label2  noniid-#label3
  do
    for beta in  0.1
    do
        for device in 'cuda:0'
        do
            for epochs in 10 20 50 100 200
            do

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=fmnist_cnn  --dataset=fmnist --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/fmnist  --other=fmnist --model=fmnist_cnn --dataset=fmnist --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=mnist_cnn  --dataset=mnist --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/mnist  --other=mnist --model=mnist_cnn --dataset=mnist --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition

            python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=svhn_cnn  --dataset=svhn --beta=$beta  --seed=$init_seed --num_users=10 --local_ep=$epochs --epochs=200 --partition $partition
            python3 experiments_dense.py  --type=kd_train --epochs=200 --lr=0.005 --batch_size 64  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/svhn  --other=svhn --model=svhn_cnn --dataset=svhn --adv=1 --beta=$beta --seed=$init_seed --num_users 10  --partition  $partition


            done
        done
    done
  done
done
#about 12days


####81 days in all
