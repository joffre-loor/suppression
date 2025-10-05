realtime_yamnet.py
python realtime_yamnet.py --device 1 --hop 0.10 --topk 8 --min_prob 0.01 --smooth 0.30 --include "" --exclude "" --no_ansi

overall_yamnet.py
python overall_yamnet.py --device 1 --half_life 30 --hop 0.10 --topk 10 --min_prob 0.01 --smooth 0.30 --force_topk --family

realtime_panns.py
python realtime_panns.py --device 1 --hop 0.10 --topk 10 --min_prob 0.01 --smooth 0.30 --force_topk --include "" --no_ansi

suppress_filter.py
python suppress_filter.py