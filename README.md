[![HitCount](http://hits.dwyl.io/srihari-humbarwadi/YOLOv1-TensorFlow20.svg)](http://hits.dwyl.io/srihari-humbarwadi/YOLOv1-TensorFlow20)
# YOLOv1-TensorFlow2.0
This is a TensorFlow2.0 implementation of the YOLOv1 paper https://arxiv.org/abs/1506.02640, with the 
following changes,
 - The feature exactor resembles the one mentioned in the YOLO9000 paper
 - Input size is changed from 448x448 to 608x608
 - The output stride is reduced from 64 to 32, to capture smaller objects
 - Used 9 boxes per grid location, the paper uses 2. [doing this did not help much]
 
## Video results
<a href="http://www.youtube.com/watch?feature=player_embedded&v=9wjTtiVUXnE " target="_blank"><img 
src="http://img.youtube.com/vi/9wjTtiVUXnE/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" 
border="10" /></a> <a href="http://www.youtube.com/watch?feature=player_embedded&v=knSVWLZa_sU " 
target="_blank"><img src="http://img.youtube.com/vi/knSVWLZa_sU/0.jpg" alt="IMAGE ALT TEXT HERE" 
width="240" height="180" border="10" /></a> 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=ZfF9SYCjxU8 " target="_blank"><img 
src="http://img.youtube.com/vi/ZfF9SYCjxU8/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" 
border="10" /></a> <a href="http://www.youtube.com/watch?feature=player_embedded&v=H8feQqaTftg " 
target="_blank"><img src="http://img.youtube.com/vi/H8feQqaTftg/0.jpg" alt="IMAGE ALT TEXT HERE" 
width="240" height="180" border="10" /></a>

## Weights
[trained weights](https://drive.google.com/file/d/1JDrBWXvNXUuuvxWvJPIMsomqXMshnhpa/view?usp=sharing)

## TODO
- [x] Change backbone to the one from YOLO9000
- [x] Use tf.distribute.MirroredStrategy for multi gpu training
- [x] Use tf.data.Dataset to implement the data input pipeline
- [ ] Add quantitative results 
- [ ] Use learning_rate schedule during training
- [ ] Add horizonal flip in data input pipeline

## Outputs
![](outputs/8986.png)
![](outputs/9740.png)
![](outputs/8534.png)
![](outputs/1415.png) 
![](outputs/9079.png)

## Training
 - The model was trained on the Berkeley Deep Drive (BDD) dataset, which has 70,000 training images and 
10000 validation images
 - The model was trained for 200 epochs with a learning_rate of 5e-4 and a batch size of 24 [8 images 
per gpu] with heavy augmentations [brightness, saturation, contrast]
 - No learning_rate schedule was followed (but was used by the authors).
