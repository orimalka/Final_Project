# Fruit Ripeness Identification


## Steps to run Code

- Clone the repository
```
git clone https://github.com/orimalka/Final_Project.git
```
- Goto the cloned folder.
```
cd Final_Project
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download extras from [link](https://drive.google.com/file/d/1OQe0HpIIptfsiWySxa9pcOt4xf_j9r84/view?usp=drive_link) and extract in "Final_Project" directory.
Extras include train data, ripeness model pkl file, yolo7 weights,
example run and run original video.
- Goto GUI directory.
```
cd gui
```
- Run the GUI with mentioned command below.
```
python run_gui.py
```
- Select the wanted file (video/image/stream) using the Browse button and click Run Prediction

Additionally you can specify a custom yolo segmentation resolution, save segmented images in a crop folder next to the output file, and show the SORT tracking traces.


- Output file will be created in the working directory with name <b>Final_Project/segmentation/runs/predict-seg/exp#/"original-video-name.mp4"</b>

### Example result






## References
- https://github.com/RizwanMunawar/yolov7-segmentation
- https://github.com/WongKinYiu/yolov7/tree/u7/seg
- https://github.com/ultralytics/yolov5

