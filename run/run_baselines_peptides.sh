for dataset in peptides_func peptides_struct
do
  for model in GCN GAT GIN SAN
  do
    config=configs/$model/${dataset}_${model}.yaml
    python run.py --cfg $config --repeat 3
  done
done
