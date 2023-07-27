# License Plate Replace

**License Plate Replace** utilizes the yolov5 and segment_anything libraries to replace an input car image's license plate with a default logo

<p float="left">
  <img src="resources/demo-images/car11.jpg?raw=true" width="40%" />
  <img src="resources/demo-images/result_replace_11.jpg?raw=true" width="40%" /> 
</p>


## Usage
```bash
python3 app.py images/car1.jpg
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

