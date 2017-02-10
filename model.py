import glob
import csv
import cv2
import numpy as np
from random import shuffle
from sklearn.utils import shuffle as shuff
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

class behavioralCloning:
    def __init__(self, datafolder, epochs=5, batchsize=256):
        self.dataFolder = datafolder
        self.imgSize = (66, 200, 3)

        self.nb_epoch = epochs
        self.nb_batch = batchsize

        # steering angle shift for offset images
        self.steering_shift = 0.25

        # steering angle shift for pixel shift (in data augmentation) (value added for every pixel that is shifted)
        self.steering_pixel_shift = 0.017

        # add this number of additional images for each images after flipping
        self.no_augment = 7

        # augmentation parameters
        self.max_rot = 5   # max rotation in degree
        self.max_shift = 50 # max pixel_shift in x and y
        self.max_brightness = 100

        # neglect training images with angle lower than
        self.neglect_angles = 0.01

        # dropout
        self.drop = 0.5 #0.5
        # l2 reg
        self.l2 = 0.0
        # initialization
        self.init = 'he_normal' # 'glorot_normal'
        # learning rate
        self.lr = 0.00001

    def run(self):
        # read Logfile and shuffle lines, split data for validation and training
        train, valid = self.splitDict(self.shuffleDict(self.readLog()), split=0.1)
        # get generators
        train_gen = self.imageGenerator(train, augment=1)
        valid_gen = self.imageGenerator(valid)
        # define model
        mdl = self.nvidiaModel()
        # fits the model on batches with real-time data augmentation:
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        mdl.compile(optimizer=opt, loss='mse')

        # checkpoint
        filepath = "model{epoch:02d}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)
        callbacks_list = [checkpoint]
        # print summary
        print(mdl.summary())
        # fit model with generators
        history = mdl.fit_generator(train_gen, samples_per_epoch=len(train['image'])*(2 *(1+self.no_augment)),
                                    nb_epoch=self.nb_epoch, validation_data=valid_gen,
                                    nb_val_samples=len(valid['image']),callbacks=callbacks_list)

        from keras.utils.visualize_util import plot
        plot(mdl, to_file='model.png', show_shapes=True, show_layer_names=False)

        # save Model
        self.saveModel(mdl)



    def imageGenerator(self, data, augment=0):
        """if augment is 1:

        flips each image and augments the dataset by self.no_augment additional images

        Augmenting uses random shifting, rotation and brightness
        """
        while True:
            if augment ==1:
                # no of original images needed
                no_original_images = self.nb_batch/(2 *(1+self.no_augment))
            else:
                no_original_images = self.nb_batch

            # get random indices
            idx = np.random.randint(len(data['image']), size=int(no_original_images))
            # define images array
            images = np.ndarray(shape=([int(no_original_images)] + list(self.imgSize)), dtype=np.uint8)
            # load original images
            for images_idx, file_idx in enumerate(idx):

                # # load, image, convert to RGB and resize
                # images[images_idx] = cv2.resize(cv2.cvtColor(cv2.imread(data['image'][file_idx]),
                #                                              cv2.COLOR_BGR2HSV)[60:140, :], (self.imgSize[1],
                #                                                                              self.imgSize[0]))
                # load, image, convert to RGB and resize
                images[images_idx] = cv2.resize(cv2.cvtColor(cv2.imread(self.dataFolder+ data['image'][file_idx]),
                                                             cv2.COLOR_BGR2HSV)[60:140, :], (self.imgSize[1],
                                                                                             self.imgSize[0]))
            angles = data['steering'][idx]
            if augment ==1:
                # flip images and append
                images, angles = self.flipImages(images, data['steering'][idx])
                # augment data
                images, angles = self.augment_data(images, angles, self.no_augment)

            # convert from HSV to RGB
            for idx, im in enumerate(images):
                images[idx] = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)


            yield shuff(images, angles)

    def augment_data(self, images, angles, iterations):
        '''iterate over data and augment every image by 'iterations' images '''
        img_indices = range(images.shape[0])
        for img_idx in img_indices:
            for iters in range(iterations):
                new_img, new_ang = self.generate_images(images[img_idx], angles[img_idx])
                images = np.vstack([images, new_img])
                angles = np.hstack([angles, new_ang])
        return images, angles

    def generate_images(self, img, ang):
        '''Apply affine transformation with random parameters as well as a contrast shift, to generate data'''
        # apply transformations
        # get matrices for affine transformation
        m_rot = self.rotate_img(img, np.random.uniform(-1.0, 1.0) * self.max_rot)
        m_shift, new_ang = self.shift_img(np.random.uniform(-1.0, 1.0) * self.max_shift,
                                          np.random.uniform(-1.0, 1.0) * self.max_shift/2, ang)
        # combine matrices
        comb_matrix = self.combineTransformationMatrices(m_rot, m_shift)
        # apply warpAffine
        out = cv2.warpAffine(img, comb_matrix, (self.imgSize[1], self.imgSize[0]))
        out = self.changeBrightness(out)

        return out.reshape(([int(1)] + list(self.imgSize))).astype(np.uint8), new_ang

    def changeBrightness(self, img):
        '''convert image to HSV and change V-value by random value. convert back to rgb'''
        image = img.astype(np.int16)
        image[: ,: ,2] = image[:, :, 2] - int(np.random.uniform(0.1, 1.0) * self.max_brightness)
        return np.clip(image, 0, 255).astype(np.uint8)


    def combineTransformationMatrices(self, *args):
        '''Combine several transformation matrices to one'''
        matrices = [np.vstack([args[i], np.array([0, 0, 1])]) for i in range(len(args))]
        ret = args[0]
        for idx in range(1, len(args)):
            ret = np.matmul(ret, matrices[idx])
        return ret[:2, :3]

    def shift_img(self, x_shift, y_shift, angle):
        '''Calculate matrix to shift image by x_shift and y_shift pixels. Use matrix with cv2.warpAffine
        Also adjust the steering angle dependent on x_shift
        '''
        return np.float32([[1, 0, x_shift], [0, 1, y_shift]]), np.clip(np.add(angle, self.steering_pixel_shift*x_shift),
                                                                       -1.0, 1.0)

    def rotate_img(self, img, deg):
        '''Calculate matrix to rotate image by deg degrees. Use matrix with cv2.warpAffine'''
        center = (img.shape[0] / 2, img.shape[1] / 2)
        return cv2.getRotationMatrix2D(center, deg, 1)

    def flipImages(self, images, angles):
        '''flip images/angles and append to according arrays'''
        im_flipped = images[:, :, ::-1, :]
        ang_flipped = np.array(angles) * (-1)
        images = np.concatenate((images, im_flipped), axis=0)
        angles = np.concatenate((np.array(angles), ang_flipped), axis=0)
        return images, angles

    def splitDict(self, dict, split=0.1):
        '''Splits the referenced data into Validation and Test Data'''
        key = list(dict)[0] # get a key
        split_idx = int(split * len(dict[key])) # get idx to split by
        train, valid = {}, {}
        for k in dict.keys():
            train[k] = np.array(dict[k])[split_idx:]
            valid[k] = np.array(dict[k])[:split_idx]
        return train, valid

    def shuffleDict(self, dict):
        '''Shuffle the lists in the dictionary, but preserve the order across keys '''
        key = list(dict)[0] # get a key
        idx = list(range(len(dict[key]))) # get length
        shuffle(idx)  # get shuffled index
        for k in dict.keys():
            dict[k] = np.array(dict[k])[idx]
        return dict

    def readLog(self):
        '''Gets the data from the Log-File and stores it in a dict'''
        file = glob.glob(self.dataFolder+'*.csv')[0] # datafile

        with open(file, mode='r',newline='') as Log:    # read datafile and store in dict self.data
            header=Log.readline()[:-1].split(',')
            data={key:[] for key in header}
            for idx, row in enumerate(csv.DictReader(Log, fieldnames=header)):
                for key in data.keys():
                    data[key].append(row[key].strip())

        # convert steering,throttle,brake,speed to floats
        for k in ['steering', 'throttle', 'brake', 'speed']:
            data[k] = [float(i) for i in data[k]]

        # get dict with keys 'image' and 'steering', where all images are used, left & right images' steering angles
        # get shifted by self.steering shift
        ret = {'image':[], 'steering':[]}
        cams = ['center', 'left', 'right']
        offset = [0, self.steering_shift, -1*self.steering_shift]
        for idx in range(3):
            ret['image'].extend(data[cams[idx]])
            ret['steering'].extend(np.add(data['steering'], offset[idx]))

        # delete all images where the absolute steering angle is smaller than self.neglect_angles
        idx = (np.abs(ret['steering'])>self.neglect_angles)
        ret['steering'] = np.array(ret['steering'])[idx]
        ret['image'] = np.array(ret['image'])[idx] 

        return ret


    def nvidiaModel(self):
        '''efine the nvidia -paper model'''
        model = Sequential()
        #normalize
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=self.imgSize, output_shape=self.imgSize))

        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu',
                                W_regularizer=l2(self.l2), input_shape=self.imgSize))
        model.add(Dropout(self.drop))

        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu',
                                W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))

        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu',
                                W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))


        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu',
                                W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))

        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu',
                                W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))


        model.add(Flatten())
        model.add(Dense(100, activation='relu', W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))
        model.add(Dense(50, activation='relu', W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))
        model.add(Dense(10, activation='relu'))#, W_regularizer=l2(self.l2)))
        model.add(Dropout(self.drop))
        model.add(Dense(1))

        return model

    def defineModel(self):
        '''defines a model similar to the nvidia model, but deeper and with maxpooling'''
        model = Sequential()
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=self.imgSize, output_shape=self.imgSize))

        model.add(Convolution2D(36, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                input_shape=self.imgSize, init=self.init))
        model.add(Convolution2D(36, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                init=self.init))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(64, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                init=self.init))
        model.add(Convolution2D(64, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                init=self.init))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(128, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                init=self.init))
        model.add(Convolution2D(128, 3, 3, border_mode="valid", activation='relu', W_regularizer=l2(self.l2),
                                init=self.init))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', W_regularizer=l2(self.l2), init=self.init))
        model.add(Dropout(self.drop))
        model.add(Dense(50, activation='relu', W_regularizer=l2(self.l2), init=self.init))
        model.add(Dropout(self.drop))
        model.add(Dense(10, activation='relu', W_regularizer=l2(self.l2), init=self.init))
        model.add(Dropout(self.drop))
        model.add(Dense(1, W_regularizer=l2(self.l2), init=self.init))

        return model

    def saveModel(self, model):
        '''Save Model'''
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to current dir")


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Runs P3 for SDCND')

    parser.add_argument('DataFolder', type=str, help='The folder containing Simulator Data '
                                                     '(IMG-Folder and driving_log.csv)')

    args = parser.parse_args()

    # assigns the command line argument (usually the video file) to video_src
    folder = args.DataFolder
    behavioralCloning(folder).run()

# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
