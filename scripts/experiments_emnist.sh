### experiments_our choose 10 classes for emnsit and split==mnist for the EMNIST in the data.py
for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in emnist
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


for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in emnist
      do
        for device in 'cuda:0'
        do
            for epochs in  200
            do
            python3 -W ignore experiments.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=fedavg \
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

### experiments_our choose 37 classes for emnsit and split==letters for the EMNIST in the data.py
for init_seed in 0 1 2
do
  for partition in emnist
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in emnist
      do
        for device in 'cuda:0'
        do
            for epochs in  200
            do
            python3 -W ignore experiments_our.py --model=simple-cn \
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


for init_seed in 0 1 2
do
  for partition in noniid-labeldir
  do
    for beta in  0.1 0.3 0.5
    do
      for dataset in emnist
      do
        for device in 'cuda:0'
        do
            for epochs in  200
            do
            python3 -W ignore experiments.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=fedavg \
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



###about 1 days
