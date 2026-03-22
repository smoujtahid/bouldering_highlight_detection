# bouldering_highlight_detection
Project to extract highlight moments from videos of bouldering.
Using MomentDETR https://github.com/jayleicn/moment_detr.git

## Install

Build the docker image
```
docker build -t momentdetr .
```

Then run the container with access to your gpu, with a Volume mount to
```
docker run --gpus all -v $(pwd):/app -it mmomentdetr bash
```

## Data exploration

Place your videos in data/
You can visualise frames or split your video into a few seconds segments with functions in script data_preparation.py

## Inference

You can run the inference on a video with script mdetr_inference.py.
Specify the path to the video, the list of queries.
The script will run MomentDETR, extract saliency score and relevant moments per query, plot them, and finally extract the highlights from the original video.

