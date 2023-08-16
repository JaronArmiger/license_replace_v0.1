import torch
from yolov5.models.yolo import Model

model = Model('./static/yolov5s.yaml')
model.load_state_dict(torch.load('./static/model.pth'))
model.eval()

car_image_path = sys.argv[1]
image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image)

results_df = results.pandas().xyxy[0].loc[0]
print(results_df)