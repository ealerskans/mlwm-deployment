#!/bin/bash
#
# Run inference on pre-trained model using global DT as input
#

# working directory
wd=$(pwd)

# git root directory
git_root_dir=/home/ea/git

PYTHON="pdm run python"

# Step 1:
#=========
# 1.1 Get the model weights from the training
saved_model=/dmidata/users/ea/neural-lam/data/saved_model/train-hi_lam-2x300-02_27_15-4034/last.ckpt

# 1.2 Get statistics from training data so this can be used to standardize the inference data

# 1.4 Get the neural-lam config file used for training
neural_lam_config=/dmidata/users/ea/neural-lam/data/config.yaml

# 1.3 Get the command (line arguemnts) used to train the model
graph_name=rect_hi3
neural_lam_command_line_args=(
    --hidden_dim 300
    --hidden_dim_grid 150
    --num_workers 6
    --precision bf16-mixed
    --batch_size 1
    --time_delta_enc_dim 32
    --processor_layers 2
    --dynamic_time_deltas
    --config_path ${neural_lam_config}
    --model hi_lam
    --graph ${graph_name}
    --val_steps_to_log 1 2 4
    --load ${saved_model}
    --epochs 80
    --ar_steps_train 1
    --lr 0.001
    --min_lr 0.001
    --val_interval 5
    --ar_steps_eval 4
    --num_nodes 2
)

# Step 2:
#=========
# 2.1 Get you data to do inference from
# 2.2 If the data is in grib format, convert it to zarr
mars_to_zarr_git_repo=https://github.com/ealerskans/mars_to_zarr.git
mars_to_zarr_branch=feature/global-dt
git clone --single-branch -b ${mars_to_zarr_branch} ${mars_to_zarr_git_repo} ${git_root_dir}/mars_to_zarr
cd ${git_root_dir}/mars_to_zarr
pdm venv create
pdm use --venv in-project
pdm install
pdm run python -m mars_to_zarr --config example.globalDT.yaml -v
cd ${wd}


# Step 3:
#=========
# 3.1 Data preparation, graph creation and neural-lam: Clone git repositories and install environments 
# 3.1.1 mllam-data-prep
mllam_data_prep_git_repo=https://github.com/ealerskans/mllam-data-prep.git
mllam_data_prep_branch=dk-case-studies-with-global-dt
git clone --single-branch -b ${mllam_data_prep_branch} ${mllam_data_prep_git_repo} ${git_root_dir}/mllam-data-prep
cd ${git_root_dir}/mllam-data-prep
pdm venv create
pdm use --venv in-project
pdm install -G latlon-domain-crop
cd ${wd}
# 3.1.2 weather-model-graphs
weather_model_graphs_git_repo=https://github.com/ealerskans/mllam-data-prep.git
weather_model_graphs_branch=building-ml-lams
git clone --single-branch -b ${weather_model_graphs_branch} ${weather_model_graphs_git_repo} ${git_root_dir}/weather-model-graphs
cd ${git_root_dir}/weather-model-graphs
pdm venv create
pdm use --venv in-project
pdm install --dev
cd ${wd}
# 3.1.3 neural-lam
neural_lam_git_repo=https://github.com/ealerskans/neural-lam.git
neural_lam_branch=neural-lam-dev-research
git clone --single-branch -b ${neural_lam_branch} ${neural_lam_git_repo} ${git_root_dir}/neural-lam
cd ${git_root_dir}/neural-lam
pdm venv create --with-pip
pdm run python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
pdm install --group dev,graph
cd ${wd}

# Step 4:
#=========
# 4.1 Create mllam-data-prep datastore
cd ${git_root_dir}/mllam-data-prep
mdp_config=/dmidata/users/ea/neural-lam/data/globalDT.20241214.yaml
${PYTHON} -m mllam_data_prep ${mdp_config}
cd ${wd}

# Step 5:
#=========
# 5.1 Create graph - Hierarchical 3-level
cd ${git_root_dir}/neural-lam
lev=3
#MND=12500
MND=1.85
${PYTHON} -m neural_lam.build_rectangular_graph \
  --config_path ${neural_lam_config} \
  --archetype hierarchical \
  --max_num_levels ${lev} \
  --mesh_node_distance ${MND} \
  --graph_name ${graph_name}
cd ${wd}

# Step 6:
#=========
# 6.1 Run inference
cd ${git_root_dir}/neural-lam
${PYTHON} -m neural_lam.train_model "${neural_lam_command_line_args[@]}" \
    --eval val \
    --save_eval_to_zarr_path /dmidata/users/ea/neural-lam/data/state_predictions.zarr

# Open questions:
# - Where does the training stat enter the equation?
# - Graph needs to match the graph used for training?
