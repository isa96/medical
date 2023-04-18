import tensorflow


class Model_resnet50_mod(tensorflow.keras.models.Sequential):
    def __init__(self):
        super(Model_resnet50_mod, self).__init__()
        self = tensorflow.keras.models.load_model('model/model4_87.h5')
        self.load_weights('model/model4_87.h5')
        print("success")

    def build_model(self):
        input_t = tensorflow.keras.Input(shape=(512, 512, 3))
        base_model = tensorflow.keras.applications.ResNet50(include_top=False,
                 weights=None,
                 input_tensor=input_t,
                 )
        
        to_res = (512, 512) #Resnet have input layer (244, 244)
        resnet50_mod = tensorflow.keras.models.Sequential()
        resnet50_mod.add(tensorflow.keras.layers.Lambda(lambda image:tensorflow.image.resize(image, to_res)))
        resnet50_mod.add(tensorflow.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu'))
        resnet50_mod.add(base_model)
        resnet50_mod.add(tensorflow.keras.layers.Conv2D(32, (3,3)))
        resnet50_mod.add(tensorflow.keras.layers.BatchNormalization())
        # resnet50_mod.add(tf.keras.layers.Dropout(0.9))
        resnet50_mod.add(tensorflow.keras.layers.Dense(4, activation='relu'))
        resnet50_mod.add(tensorflow.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'))
        resnet50_mod.add(tensorflow.keras.layers.Flatten())
        resnet50_mod.add(tensorflow.keras.layers.Dense(4, activation='softmax'))
        return resnet50_mod
        

#class Model_lu_Net():
#    def __init__():
#        model = tensorflow.keras.models.load_model()
    