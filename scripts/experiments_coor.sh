for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in 0.01 0.05 0.1 0.3 0.5 1
    do
      for dataset in fmnist
      do
        for device in 'cuda:0'
        do
          for epochs in  200
          do
            for coor in 0.99 0.9999
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
              --coor=$coor

            done
          done
        done
      done
    done
  done
done

for init_seed in 0 1 2
do
  for partition in noniid-#label1  noniid-#label2  noniid-#label3
  do
    for beta in  0.1
    do
      for dataset in fmnist
      do
        for device in 'cuda:0'
        do
          for epochs in 200
          do
            for coor in 0.99 0.9999
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
              --coor=$coor
              done
          done
        done
      done
    done
  done
done

## about 2 days

