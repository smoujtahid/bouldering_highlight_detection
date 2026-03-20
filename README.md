# bouldering_highlight_detection


## Install

Build the docker image
```
docker build -t momentdetr
```

Then run the container with access to your gpu, with a Volume mount to
```
docker run --gpus all -v $(pwd):/app -it mmomentdetr bash
```

## Data exploration

Place your videos in data/
You can visualise frames or split your video into a few seconds segments with functions in script data_preparation.py



