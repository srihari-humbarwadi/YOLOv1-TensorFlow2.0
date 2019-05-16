#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import json
import cv2
import os


# In[2]:


H, W = 608, 608
grid_size = [H//32, W//32]
nboxes = 9
classes = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck']
class_map = {k:idx for idx,k in enumerate(classes)}
nclasses = len(class_map)
output_shape = grid_size + [nboxes*5 + nclasses]


# In[3]:


def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

    # calculate the left up point & right down point
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)    

offset = []
offset_tran = []
for i in range(grid_size[0]):
    row = []
    row_trans = []
    for j in range(grid_size[0]):
        row.append(j)
        row_trans.append(i)
    offset.append(row)
    offset_tran.append(row_trans)
offset = np.tile(np.array(offset)[None, :, :, None], reps=[1,1,1,nboxes])
offset_tran = np.tile(np.array(offset_tran)[None, :, :, None], reps=[1,1,1,nboxes])

offset = tf.constant(offset, dtype=tf.float32)
offset_tran = tf.constant(offset_tran, dtype=tf.float32)

def Yolo_Loss(y_true=None, y_pred=None, eval=False):
    pred_obj_conf = y_pred[:,:,:,:nboxes]
    pred_box_classes = y_pred[:,:,:,5*nboxes:]
    pred_box_offset_coord = y_pred[:,:,:, nboxes:5*nboxes]
    pred_box_offset_coord = tf.reshape(pred_box_offset_coord, shape=[-1, grid_size[0], grid_size[0], nboxes, 4])
    pred_box_normalized_coord = tf.stack([(pred_box_offset_coord[:,:,:,:,0] + offset)/grid_size[0],
                                         (pred_box_offset_coord[:,:,:,:,1] + offset_tran)/grid_size[0],
                                         tf.square(pred_box_offset_coord[:,:,:,:,2]),
                                         tf.square(pred_box_offset_coord[:,:,:,:,3])], axis=-1)
    if eval:
        return pred_obj_conf, pred_box_classes, pred_box_normalized_coord
    target_obj_conf = y_true[:,:,:,:1]
    target_box_classes = y_true[:,:,:,5:]
    target_box_coord = y_true[:,:,:,1:5]
    target_box_coord = tf.reshape(target_box_coord, shape=[-1, grid_size[0], grid_size[1], 1, 4])
    target_box_coord = tf.tile(target_box_coord, multiples=[1,1,1,nboxes,1])
    target_box_normalized_coord = target_box_coord / H
    target_box_offset_coord = tf.stack([target_box_normalized_coord[:,:,:,:,0]*grid_size[0] - offset,
                                        target_box_normalized_coord[:,:,:,:,1]*grid_size[0] - offset_tran,
                                        tf.sqrt(target_box_normalized_coord[:,:,:,:,2]),
                                        tf.sqrt(target_box_normalized_coord[:,:,:,:,3])], axis=-1)

    pred_ious = compute_iou(target_box_normalized_coord, pred_box_normalized_coord)
    predictor_mask = tf.reduce_max(pred_ious, axis=3, keepdims=True)
    predictor_mask = tf.cast(pred_ious>=predictor_mask, tf.float32) * target_obj_conf
    noobj_mask = tf.ones_like(predictor_mask) - predictor_mask

    # Computing the class loss
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_obj_conf*(target_box_classes - pred_box_classes)), axis=[1, 2, 3]))

    # computing the confidence loss
    obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask*(pred_obj_conf - pred_ious)), axis=[1, 2, 3]))
    noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask*(pred_obj_conf)), axis=[1, 2, 3]))

    # computing the localization loss
    predictor_mask = predictor_mask[:,:,:,:,None]
    loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask*(target_box_offset_coord - pred_box_offset_coord)), axis=[1, 2, 3]))

    loss = 10 * loc_loss + obj_loss + 0.1 * noobj_loss + 0.5 * class_loss
    return loss


# In[4]:


train_images = sorted(glob('/home/mia/backup/research/autonomous_driving/BDD/bdd100k/images/100k/train/*'))
train_labels = sorted(glob('/home/mia/backup/research/autonomous_driving/BDD/bdd100k/labels/100k/train/*'))

val_images = sorted(glob('/home/mia/backup/research/autonomous_driving/BDD/bdd100k/images/100k/val/*'))
val_labels = sorted(glob('/home/mia/backup/research/autonomous_driving/BDD/bdd100k/labels/100k/val/*'))

batch_size = 24
train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size


# In[5]:


def get_label(label_path, orig_h=720, orig_w=1280):
    label = np.zeros(shape=[*grid_size, 5+nclasses])
    with open(label_path, 'r') as f:
        temp = json.load(f)
    for obj in temp['frames'][0]['objects']:
        if 'box2d' in obj:
            x1 = obj['box2d']['x1'] * W / orig_w
            y1 = obj['box2d']['y1'] * H / orig_h
            x2 = obj['box2d']['x2'] * W / orig_w
            y2 = obj['box2d']['y2'] * H / orig_h
            x = (x2 + x1) / 2
            y = (y2 + y1) / 2
            w = x2 - x1
            h = y2 - y1
            category_id = class_map[obj['category']]
            class_vector = np.zeros((nclasses, ))
            class_vector[category_id] = 1
            label_vector = [1, x, y, w, h, *class_vector]
            grid_x = int(x / W * grid_size[1])
            grid_y = int(y / H * grid_size[0])
            try:
                label[grid_y, grid_x] = label_vector
            except:
                continue
    return label


# In[70]:


train_label_vectors = np.zeros(shape=[len(train_images), *grid_size, 5+nclasses])
for i, img in tqdm(enumerate(train_images)):
    fname = img.split('/')[-1].split('.')[0] + '.json'
    label_path = '/home/mia/backup/research/autonomous_driving/BDD/bdd100k/labels/100k/train/' + fname
    train_label_vectors[i] = get_label(label_path)

val_label_vectors = np.zeros(shape=[len(val_images), *grid_size, 5+nclasses])
for i, img in tqdm(enumerate(val_images)):
    fname = img.split('/')[-1].split('.')[0] + '.json'
    label_path = '/home/mia/backup/research/autonomous_driving/BDD/bdd100k/labels/100k/val/' + fname
    val_label_vectors[i] = get_label(label_path)                            


# In[5]:


def get_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(tf.image.resize(img, size=[H, W]), dtype=tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    img /= 127.5
    img -= 1.
    return img

def load_data(image_path, labels):
    flip = tf.cast(tf.random.uniform(shape=[1,], minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool).numpy()[0]
    return get_image(image_path), flip_labels(labels)


# In[5]:


def conv_block(x, n_filters, size, pool=False):
    x = Conv2D(filters=n_filters,
               kernel_size=size,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    if pool:
        x = MaxPool2D(pool_size=2)(x)
    return x

input_layer = Input(shape=(H, W, 3))
x = conv_block(input_layer, 32, 3, pool=True)
x = conv_block(x, 64, 3, pool=True)
x = conv_block(x, 128, 3)
x = conv_block(x, 64, 1)
x = conv_block(x, 128, 3, pool=True)
x = conv_block(x, 256, 3)
x = conv_block(x, 128, 1)
x = conv_block(x, 256, 3, pool=True)
x = conv_block(x, 512, 3)
x = conv_block(x, 256, 1)
x = conv_block(x, 512, 3)
x = conv_block(x, 256, 1)

skip = Lambda(lambda tensor : tf.nn.space_to_depth(tensor, block_size=2))(x)

x = conv_block(x, 512, 3, pool=True)
x = conv_block(x, 1024, 3)
x = conv_block(x, 512, 1)
x = conv_block(x, 1024, 3)
x = conv_block(x, 512, 1)
x = conv_block(x, 1024, 3)
x = conv_block(x, 1024, 3)
x = conv_block(x, 1024, 3)
x = concatenate([x, skip])
x = conv_block(x, 1024, 3)

output_layer = Conv2D(output_shape[-1], kernel_size=1)(x)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Yolo')
model.load_weights('model/weights.h5')


# In[8]:


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_label_vectors))
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(map_func=load_data, 
                                                   batch_size=batch_size, 
                                                   num_parallel_calls=64,
                                                   drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(-1)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_label_vectors))
val_dataset = val_dataset.shuffle(buffer_size=256)
val_dataset = val_dataset.apply(tf.data.experimental.map_and_batch(map_func=load_data,
                                                   batch_size=batch_size, 
                                                   num_parallel_calls=64,
                                                   drop_remainder=True))
val_dataset = train_dataset.repeat()
val_dataset = train_dataset.prefetch(-1)


# In[7]:


# !rm -r model/*
# callbacks = [tf.keras.callbacks.ModelCheckpoint('model/weights.h5', save_best_only=True, save_weights_only=True)]
# model.fit(train_dataset,
#          steps_per_epoch=train_steps,
#          epochs=300, 
#          validation_data=val_dataset,
#          validation_steps=val_steps, 
#          callbacks=callbacks)


# In[75]:


label_map = {v:k for k,v in class_map.items()}

def read_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(tf.image.resize(img, size=[H, W]), dtype=tf.float32)
    return img.numpy()

def visualize(img, pred, iou=0.5, score_threshold=0.5, label_map={}, orig_h=720, orig_w=1280, resize=True, video=True):
    predict_object, predict_class, predict_normalized_box  = Yolo_Loss(pred, pred, eval=True)
    predict_object, predict_class, predict_normalized_box = predict_object[0], predict_class[0], predict_normalized_box[0].numpy()
    if not resize:
        predict_normalized_box = predict_normalized_box * H
    else:
        img = cv2.resize(img, (orig_w, orig_h))
        predict_normalized_box[...,0] = predict_normalized_box[...,0] * orig_w
        predict_normalized_box[...,1] = predict_normalized_box[...,1] * orig_h
        predict_normalized_box[...,2] = predict_normalized_box[...,2] * orig_w
        predict_normalized_box[...,3] = predict_normalized_box[...,3] * orig_h 
        H, W = orig_h, orig_w
    predict_normalized_box = np.stack([predict_normalized_box[:,:,:,0]-0.5*predict_normalized_box[:,:,:,2],
                                       predict_normalized_box[:,:,:,1]-0.5*predict_normalized_box[:,:,:,3],
                                       predict_normalized_box[:,:,:,0]+0.5*predict_normalized_box[:,:,:,2],
                                       predict_normalized_box[:,:,:,1]+0.5*predict_normalized_box[:,:,:,3]], axis=-1)
    predict_object = np.reshape(predict_object, newshape=[-1, 1])
    predict_class = np.reshape(predict_class, newshape=[-1, 1, nclasses])
    predict_class = np.tile(predict_class, reps=[1, nboxes, 1])
    predict_class = np.reshape(predict_class, newshape=[-1, nclasses])
    predict_class_conf = np.max(predict_class, axis=-1)[:, None]
    predict_class_idx = np.argmax(predict_class, axis=-1)[:, None]
    scores = np.max(predict_class_conf * predict_object, axis=-1)
    predict_normalized_box = np.reshape(predict_normalized_box, newshape=[-1, 4])
    indices = tf.image.non_max_suppression(predict_normalized_box, scores=scores, iou_threshold=iou, score_threshold=score_threshold, max_output_size=100).numpy()
    
    boxes = np.int32(predict_normalized_box[indices])
    probs = scores[indices]
    category = np.reshape(predict_class_idx[indices], newshape=[-1])

    img = np.uint8(img)
    for idx, box in enumerate(boxes):
        color_map = [(161, 244, 171), (179, 165, 215), (0, 0, 255), (144, 158, 187), (255, 0, 150), (239, 100, 124), (188, 128, 221), (156, 171, 161), (131, 180, 204), (187, 208, 123)]
        bcolor = color_map[category[idx]]
#         label_w = 150
#         label_h = 45

#         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), bcolor, 8)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), bcolor, 2)
        
        cv2.rectangle(img, (box[0], box[1]), (box[0]+W//10, box[1]-H//32), (0, 255, 0), -1)
#         cv2.rectangle(img, (box[0], box[1]), (box[0]+label_w, box[1]-label_h), (0, 255, 0), -1)
        
#         cv2.putText(img, f'{label_map[category[idx]]}', (box[0]+5, box[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10,10,10), 4, lineType=cv2.LINE_8)
        cv2.putText(img, f'{label_map[category[idx]]}', (box[0]+5, box[1]-3), cv2.FONT_HERSHEY_SIMPLEX, .7, (10,10,10), 2, lineType=cv2.LINE_AA)

    if video:
        return img
    plt.figure(figsize=(16, 9))
    plt.imshow(img)
    plt.title('output')
    plt.axis('off')
    plt.savefig(f'outputs/{np.random.randint(0, 10000)}.png')
    plt.close()
    
def pipeline(frame):
    h,w = frame.shape[:2]
    disp = frame.copy()
    frame = np.float32(cv2.resize(frame, (W,H)))
    frame /= 255.
    pred = model.predict(frame[None, ...])
    output = visualize(disp, pred, iou=0.5, score_threshold=0.1, label_map=label_map, orig_h=h, orig_w=w)    
    return output


# In[71]:


# color_map = {i:tuple(v) for i, v in enumerate(np.random.randint(100, 255, size=[10, 3]))}
# color_map = list(color_map.values())


# In[76]:


for i in tqdm(np.random.randint(0, 10000, 500)):
    img = val_images[i]
    img = read_img(img)
    disp = img.copy()
    img /= 255.
    pred = model.predict(img[None, ...])
    x = visualize(disp, pred, iou=0.5, score_threshold=0.4, label_map=label_map, video=False)


# In[62]:


clip = VideoFileClip('../../4k_driving/toronto.webm')
res = clip.fl_image(pipeline)
res.write_videofile('darknet_toronto.mp4', audio=False, threads=16)


# In[ ]:


clip = VideoFileClip('../../4k_driving/vancouver.webm')
res = clip.fl_image(pipeline)
res.write_videofile('darknet_vancouver.mp4', audio=False, threads=16)


# In[ ]:


clip = VideoFileClip('../../4k_driving/seattle.webm')
res = clip.fl_image(pipeline)
res.write_videofile('darknet_seattle.mp4', audio=False, threads=16)


# In[ ]:


clip = VideoFileClip('../../4k_driving/karol.mp4')
res = clip.fl_image(pipeline)
res.write_videofile('darknet_karol.mp4', audio=False, threads=16)


# In[ ]:




