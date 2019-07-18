## Python and Toolkit Version
	Python:      '3.5.2'
	numpy:       '1.13.1'
	tensorflow:  '1.4'

## Preprocess example script
[Gist link].(https://gist.github.com/acetylSv/1a233574234cb4188629c90b49bee1da)

## Training                                                            
	python3 gan_train.py

## Inference
	python3 gan_test.py

## Hyperparameter
	All hyperparameter tuning could be done in hyperparams.py.

## Experiments and Samples
	Currently done on VCTK dataset and can be found in results/

### TODO

- [ ] Remove per-speaker information from Graph eg. f0, mcep_mean...
- [ ] Neater data preprocess
