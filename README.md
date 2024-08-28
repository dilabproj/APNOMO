# Description
Work Name: **early time series Anomaly Prediction with Neighbor Over-sampling and Multi-objective Optimization (APNOMO)**

# Files
- train.py		feature extractor training
- test.py		predictor training
- model.py		model architecture
- utils.py		utilized functions
- early_stop.py		Early stopping rule & MOO
- dataset.py		dataset related process
- our2.yml		environment (Please use miniconda to create env first)

# Run Process
```
conda activate our2

## training
python train.py --dataset $dataset_name --type $type
python test.py --dataset $dataset_name --type $type
```
Argument: \
$dataset_name: {01hr2, 03hr2, 06hr2, 12hr2, 24hr2} \
Over-sampling: $type =“_upXX%” \
e.g., $dataset_name = 06hr2, $type =“_up30%” 
Time interval 6 hours with oversampling rate 30%

# Parameters
![image](https://github.com/user-attachments/assets/96b36a6f-58a4-4e6c-961f-5d4cbc6f6f88)
![image](https://github.com/user-attachments/assets/03ad1207-3061-4f75-8fe0-5ce64033bcc7)



