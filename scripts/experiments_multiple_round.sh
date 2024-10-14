for init_seed in 0 1 2
do
  for partition in noniid-labeldir 
  do
    for beta in 0.5
    do
      for dataset in mnist 
      do
        for device in 'cuda:0'
        do
            for epochs in 10
            do
            python3 -W ignore experiments_our.py --model=simple-cnn \
              --dataset=$dataset \
              --alg=our \
              --lr=0.01 \
              --batch-size=64 \
              --epochs=$epochs \
              --n_parties=10 \
              --rho=0.9 \
              --comm_round=50 \
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
    for beta in  0.5 
    do
      for dataset in mnist
      do
        for device in 'cuda:0'
        do
    	  for epochs in 10 
          do
           
            for alg in fednova scaffold fedavg fedprox
            do
              python3 experiments.py --model=simple-cnn \
                --dataset=$dataset \
                --alg=$alg \
                --lr=0.01 \
                --batch-size=64 \
                --epochs=$epochs  \
                --n_parties=10 \
                --rho=0.9 \
                --comm_round=50 \
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

## you need to change the code a little bit for our (one round) + fedavg
#### 1 day
