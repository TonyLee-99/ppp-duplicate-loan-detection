PY ?= python
all:
	cd src && $(PY) 01_load.py && $(PY) 02_standardize.py && $(PY) 03_blocking.py && $(PY) 04_features.py \
	  && $(PY) 06_splink_baseline.py && $(PY) 07_train_matcher.py && $(PY) 08_cluster_flag.py && $(PY) 09_report.py
# 05_sample_for_labeling.py / 05b / 05c draw pairs for (re)labeling; labels/labels.csv holds the existing labels.
