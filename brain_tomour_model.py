import tensorflow as tf 
import os
import cv2
import imghdr 
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('CPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu , True)
data_dir  = '/home/bhalewar/Downloads/ brain_tumber/Training'
image_exts = ['jpeg' , 'jpg' , 'bmp' , 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class , image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'image not in ext list {image_path}')
                os.remove(image_path)
        except Exception as e :
            print(f'error{e} with image {image_path}')
img = cv2.imread('/home/bhalewar/Downloads/ brain_tumber/Training/glioma/Tr-gl_0010.jpg')
plt.imshow(img)
plt.show()
# load data set 
data = tf.keras.utils.image_dataset_from_directory(data_dir)
print(data.class_names)
class_names = data.class_names
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
lass_names = data.class_names
print("Class names:", class_names)
for i, name in enumerate(class_names):
    print(f"{i} → {name}")
fit, ax = plt.subplots(ncols=4 , figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
data = data.map(lambda x,y: (x/255 , y))
scaled_iterator = data.as_numpy_iterator()
tbatch = scaled_iterator.next()
fit, ax = plt.subplots(ncols=4 , figsize=(20,20))
for idx, img in enumerate(tbatch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(tbatch[1][idx])
# train test set 
train_sixe = int(len(data)*.8)
val_size = int(len(data)*.2)+1
test_data_dir = '/home/bhalewar/Downloads/ brain_tumber/Testing'
test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir)
class_names_test  = test_data.class_names
class_names = test_data.class_names
print("Class names:", class_names)
for i, name in enumerate(class_names):
    print(f"{i} → {name}")
# preproceing test data set 
test_data_iterator = test_data.as_numpy_iterator()
test_batch = test_data_iterator.next()
test_data =  test_data.map(lambda x,y: (x/255 , y))
test_scaled_iterator = test_data.as_numpy_iterator()
test_batch = test_scaled_iterator.next()
fit, ax = plt.subplots(ncols=4 , figsize=(20,20))
for idx, img in enumerate(test_batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(test_batch[1][idx])
train_sixe + val_size == len(data)
test_size = int(len(test_data))
train = data.take(train_sixe)
val = data.skip(train_sixe).take(val_size)
test = test_data.take(test_size)
print(len(train) + len(val) == len(data))
print(len(test) == len(test_data))
#bulding Nural Network 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten , Dropout
model = Sequential()
model.add(Conv2D(16,(3,3), 1 , activation='relu' , input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64 , (3,3) , 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32 , (3,3) , 1, activation='relu'))
model.add(MaxPooling2D())
# model.add(Conv2D(16 , (3,3) , strides=(1,1), activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4 , activation='softmax')) # they are 4 classes 
model.compile('adam' , loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary() # optional to run these will proved the parameter and other detailes of model 
# call backs 
logdir = '/home/bhalewar/logs'
tensorbord_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# model traning 
hist = model.fit(train , epochs=20 , validation_data=val , callbacks=[tensorbord_callback])
# evaluate 
test_loss, test_acc = model.evaluate(test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
# model  performace visualation 
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal' , label='Loss')
plt.plot(hist.history['val_loss'], color='red' , label='Val_loss')
fig.suptitle('LOSS' , fontsize=20)
plt.legend(loc='upper right')
plt.show()

fig = plt.Figure()
plt.plot(hist.history['loss'], color='red' , label='LOSS')
plt.plot(hist.history['accuracy'], color='green' , label='ACCURACY')
fig.suptitle('overall result' , fontsize=20)
plt.legend(loc="lower left")
plt.show()

fig = plt.Figure()
plt.plot(hist.history['val_loss'], color='red' , label='val_LOSS')
plt.plot(hist.history['val_accuracy'], color='green' , label='val_ACCURACY')
fig.suptitle('real overall result' , fontsize=20)
plt.legend(loc="lower left")
plt.show()

from tensorflow.keras.metrics import Precision, Recall, Accuracy
import numpy as np

pre = Precision()
re = Recall()
acc = Accuracy()
for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    yhat_classes = np.argmax(yhat, axis=1)
    pre.update_state(y, yhat_classes)
    re.update_state(y, yhat_classes)
    acc.update_state(y, yhat_classes)

print(f"precision: {pre.result().numpy()}\n recall: {re.result().numpy()} \n accuracy: {acc.result().numpy()}")

# testing random images 
class_names

img = cv2.imread('/home/bhalewar/Downloads/bb.jpg')
plt.imshow(img)
plt.show()

re  = tf.image.resize(img , (256 , 256))
plt.imshow(re.numpy().astype(int))
plt.show()

ynew = model.predict(np.expand_dims(re/255,0))

ynew_classes = np.argmax(ynew, axis=1)
ynew_classes
match(ynew_classes):
    case 0:
        print(f'glioma with confidence {np.max(ynew)*100}')
    case 1:
        print(f'meningioma with confidence {np.max(ynew)*100}')
    case 2 :
        print(f' no tumor with confidence {np.max(ynew)*100}')
    case 3:
        print(f'pituitary with confidence {np.max(ynew)*100}')
    case _:
        print(f'unknown with confidence {np.max(ynew)*100}')


# save the model 
from tensorflow.keras.models import load_model
model.save(os.path.join('models' , 'brain tumber model.h5'))

# re load model for test

loadmodel = load_model(os.path.join('models' , 'brain tumber model.h5'))
pr = loadmodel.predict(np.expand_dims(re/255,0))

yn  = np.argmax(pr, axis=1)
print(yn)
confi = np.max(pr)
print(confi*100)
# And all set out model is perfectly working 
