for dataset in cora citeseer pubmed
do
  for model in GCN GAT GIN
  do
    config=configs/$model/${dataset}_${model}.yaml
    python run.py --cfg $config --repeat 3
  done
done
