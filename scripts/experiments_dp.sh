for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in fmnist
      do
        for device in 'cuda:0'
        do
            for epochs in  200
            do
            python3 -W ignore experiments_our_dp.py --model=simple-cnn \
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


###using the utils_dp.py replace utils.py
for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in fmnist
      do
        for device in 'cuda:0'
        do
            for epochs in  200
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
###laplace_distribution = Laplace(0,0.125) laplace_distribution = Laplace(0,0.2) laplace_distribution = Laplace(0,0.333)
###for 0.125 0.2 0.333 run three times!


###about 1 days
