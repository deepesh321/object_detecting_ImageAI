from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
    '/home/deepesh/object_detecting_ImageAI/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False)
detections = detector.detectCustomObjectsFromImage(
    input_image='zebra1.jpg', output_image_path='/home/deepesh/object_detecting_ImageAI/output_image1.png', custom_objects=custom_objects, minimum_percentage_probability=65)
for obj in detections:
    print(obj['name']+':'+obj['percentage_probability'])
