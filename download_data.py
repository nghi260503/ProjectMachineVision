from roboflow import Roboflow
rf = Roboflow(api_key="aXFRu0Sw8nHiGJwcMwqW")
project = rf.workspace("qua-mn-nh").project("heatmap-batminton")
version = project.version(1)
dataset = version.download("yolov8")