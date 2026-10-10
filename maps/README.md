# Що лежить у `maps/`

Усе тут будує один скрипт `build_all_prob_maps.sh` (він викликає
`src/occupancy_3d_map.py`). Перезапуск повністю перезаписує теку набору.

```
maps/
  all_maps_summary.csv              зведена таблиця метрик по всіх прогонах
  <набір>/<vslam>_<fde>/
      naive_grid.npz                базова карта: усі спостереження з вагою 1,
                                    без калібрування глибини (для порівняння)
      adaptive_grid.npz             ймовірнісна карта: вага довіри на кожен
                                    піксель + локальне калібрування глибини
                                    по розріджених точках VSLAM
      adaptive_grid.html            перегляд у браузері (Plotly)
      prob2d/prob_<мітка>.png       2D карта ймовірності зайнятості для КОЖНОГО
                                    обробленого кадру: P = sigmoid(l_occ * w)
                                    по пікселях після всіх перетворень
                                    (мережа/таблиця -> калібрування -> маски).
                                    Пусті пікселі = відкинуті, у карту не йдуть.
```

Наборів сім: `euroc` (№1, Isaac Sim) і `run1`–`run6` (№2–№7, Cosys-AirSim).
Моделі глибини: `zoedepth`, `depthanythingv2`, `fastdepth`.
VSLAM: `dpvo` для всіх наборів, `orbslam3` лише для `euroc` (на AirSim він губить
трекінг після перших від 26 до 106 кадрів, карти з нього там не буде).

Таблиця корекції глибини для кожного набору навчена БЕЗ цього набору
(`refiners/loo_<набір>_<fde>.npz`), тому набір, на якому будується карта,
у навчанні не брав участі.

Що у `.npz`: `log_odds` (сітка log-odds), `bounds_min`, `voxel_resolution`.
Ймовірність зайнятості вокселя: `P = 1 / (1 + exp(-log_odds))`.

## Перегляд

```bash
cd src && source ../.venv/bin/activate
python3 view_3d_occupancy_map.py ../maps/euroc/dpvo_zoedepth/adaptive_grid.npz \
    --show_free --trajectory ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor/state_groundtruth_estimate0/data.csv
```

У WSL відкривати так: `explorer.exe maps\euroc\dpvo_zoedepth\adaptive_grid.html`.
2D карти по кадрах дивитись просто як картинки у `prob2d/`.

`prob2d/` і `logs/` не в git (`.gitignore`), бо відтворюються скриптом.
