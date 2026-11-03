# MasterThesis
Safety filter for autonomous drone flight

export DATASET_PATH=~/MasterThesis/datasets/euroc/2026-04-01_12-13-18_000_quadrotor
export DATASET_PATH=~/MasterThesis/datasets/run1
export DATASET_PATH=~/MasterThesis/datasets/run2
export DATASET_PATH=~/MasterThesis/datasets/run3
export DATASET_PATH=~/MasterThesis/datasets/run4
export DATASET_PATH=~/MasterThesis/datasets/run5
export DATASET_PATH=~/MasterThesis/datasets/run6

docker compose build zoedepth depthanythingv2 fastdepth   
docker compose build dpvo                                  
docker compose build orbslam3                              


