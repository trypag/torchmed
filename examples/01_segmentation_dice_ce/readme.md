## Code organization

- datasets : converts input data into iterable datasets for training and inference.
  - `training.py` : build a dataset for inference.
  - `inference.py` : build the datasets for training and validation.
  - `mappings.py` : mapping of image labels.
- `architecture.py` : architecture of the CNN.
- `inference_canvas.py` : inference and metrics evaluation for the test dataset.
- `segment.py` : script for segmenting images based on a model.
- `train_segment.py` : training + segmentation of test dataset.
- `train.py` : training script.

The output folder will contain :

- figures plotting the various segmentation metrics and losses.
- logs of metrics and losses for each iteration on train and validation.
- `checkpoint.pth.tar` : last epoch model's parameters.
- `model_best_dice.pth.tar` : best performing model's parameters.
