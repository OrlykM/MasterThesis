# MasterThesis
Safety filter for autonomous drone flight

export DATASET_PATH=~/MasterThesis/datasets/euroc/euroc_runs/2026-04-01_12-13-18_000_quadrotor
docker compose build zoedepth depthanythingv2 fastdepth   
docker compose build dpvo                                  
docker compose build orbslam3                              