# License Plate Replace

**License Plate Replace** utilizes the yolov5 and segment_anything libraries to replace an input car image's license plate with a default logo

<div style="display:flex; justify-content:space-between">
  <div>
    <h4>original</h4>
    <img src="resources/demo-images/car8.jpg?raw=true" />
  </div>
  
  <div>
    <h4>default logo</h4>
    <img src="resources/demo-images/rep_08_default.jpg?raw=true"  />
  </div>
  <div>
    <h4>custom logo (passed as second command line arg)</h4>
    <img src="resources/demo-images/rep_08_arg.jpg?raw=true" />
  </div>
</div>


## Usage
car image path required
logo image path optional (if none, default logo is used)
```bash
python3 app.py /path/to/car_image [/path/to/logo_image]
```

modified images are output to __static/results/__ directory


## Setup
Clone **license_replacev0.1** repo
```bash
git clone git@github.com:JaronArmiger/license_replace_v0.1.git
```

(OPTIONAL) set up python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

Install root dependencies
```bash
pip3 install -r requirements.txt
```

Add yolov5 to project (yolov5's dependencies are already included in root requirements.txt)
```bash
git clone https://github.com/ultralytics/yolov5
```

Add segment_anything to project
```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Download checkpoint for segment_anything
```bash
curl -o ./static/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

```Dockerfile
# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM public.ecr.aws/docker/library/python:buster as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR
ENV USE_NNPACK=0
ENV USE_CUDA=0

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  ffmpeg \
  libxext6 \
  libsm6

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the function's dependencies
RUN pip install \
  --target ${FUNCTION_DIR} \
  -r requirements.txt


FROM public.ecr.aws/docker/library/python:buster

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]```

