# WP2 - Knowledge Graph Drievn Reinforcement Learning Based Explainable Recommender System
This approach mainly have 5 modules, preprocessing and kg creation, generating kg embedding, train reinforcement learning agent, recommending items for customers and path generation and evaluation.

## Datasets
We are using Fidelity RecSys Challenge sample dataset v3. This dataset has three csv files - articles.csv, user_v3.csv and Rec_Sys_sample_train_data_v3.csv in the folder named Fidelity_RecSys_Challenge_Sample_Dataset_v3. The code will divide the Rec_Sys_sample_train_data_v3.csv in train and test data. 
You need to upload these three data files to Amazon S3 and replace the path in the code with your own path. 
We have already uploaded these data file to S3. If you can access the existing data path that is there in the code then there is no need to change the path.
Please upload the RL_based_Recommender_System folder as it is to the Amazon Sagemaker and run the three .ipynb files as described below.


## How to run the code 

All the code need to be run on Amazon Sagemaker. Run the below three .ipynb files in the same order as they are listed.

1. Preprocess the data:

run the processing.ipynb file

2. KG creation, embeddings generation, model building:

run the pgpr.ipynb file

3. Evaluation and Path generation:

run evaluation.ipynb


## Requirements
- Python >= 3.6
- PyTorch = 1.0


## Directory layout

  .
    ├── ...
    ├── RL_based_Recommender_System     					# Project Directory
    │   ├── data ──               							# Folder containing Challenge_Dataset folder
    │   ├          ├── Challenge_Dataset   					# Folder to store the preprocessed data files that code will generate
	│   ├── Fidelity_RecSys_Challenge_Sample_Dataset_v3     # Folder containing articles.csv, user_v3.csv and Rec_Sys_sample_train_data_v3.csv files
    │   ├── data_utils.py           						# Custom utility functions
    │   ├── evluation.ipynb           						# Evaluation and path generation
    │   ├── kg_env.py            							# Train RL agent
    │   ├── knowledge_graph.py      						# Create Knowledge Graph 
    │   ├── pgpr.ipynb           							# KG creation, generating kg embedding, train reinforcement learning agent and prediction    
    │   ├── processing.ipynb     							# Preprocess the Fidelity data and create required data files in .txt format
    │   ├── transe_model.py        							# Generate KG embedding
    │   ├── utils.py    									# Custom utility functions
    │   ├── requirements.txt        						# Project dependency
    │   └── README.md               						# Project description
    └── ...


## Parameter Settings
For best results on dataset v3 use below parameter values:
learning rate = 0.0001
number of epoch = 10