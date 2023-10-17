for dataset in NCI1 PROTEINS PTC_MR DD MUTAG REDDIT-BINARY IMDB-BINARY IMDB-MULTI
do
for mode in 1 2 3
do
for seed in 0 1 2 3 4
do
python sgncl.py --seed $seed --dataset $dataset --mode $mode
done
done
done