# YOLOv1-TensorFlow2.0
This is a TensorFlow2.0 implementation of the YOLOv1 paper https://arxiv.org/abs/1506.02640, with the following changes,
 - The feature exactor resembles the one mentioned in the YOLO9000 paper
 - Input size is changed from 448x448 to 608x608
 - The output stride is reduced from 64 to 32, to capture smaller objects
 - Used 9 boxes per grid location, the paper uses 2. [doing this did not help much]
