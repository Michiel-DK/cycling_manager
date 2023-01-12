from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Input, Conv2D, Flatten, MaxPooling2D, Masking, LayerNormalization, RepeatVector, Concatenate, Rescaling
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import Precision
from tensorflow.keras import Model


def resnet_model(images, train_resnet=False):    
    # $CHALLENGIFY_BEGIN
    
    model = ResNet50(weights="imagenet", include_top=False, input_shape=images.shape[2:])
    
    if train_resnet == False:
        model.trainable = False
    
    # $CHALLENGIFY_END
    
    return model  

def image_model(images, output=False, resnet=True, train_resnet=False):
    
    if not resnet:
        x = TimeDistributed(Rescaling(1./255, input_shape=(150,300,3)))(images)
        x = TimeDistributed(Conv2D(16, kernel_size=10, activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(3))(x)
        x = TimeDistributed(Conv2D(32, kernel_size=10, activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(3))(x)
        x = TimeDistributed(Conv2D(16, kernel_size=8, activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(3))(x)
    
    else:
        resnet = resnet_model(images, train_resnet=train_resnet)
        x = TimeDistributed(resnet)(images)
        x = TimeDistributed(Flatten())(x)
        
    x = TimeDistributed(Dense(200, activation='tanh'))(x)
    x = TimeDistributed(Dense(30, activation='tanh'))(x)
    y = TimeDistributed(Dense(10, activation='tanh'))(x)
    y = TimeDistributed(Dense(1, activation='linear'))(y)
    
    if output:
        print(y)
        return y
    else:
        return x

def encoder(encoder_features, encoder_image_features):
    y = Masking(mask_value = -1000.)(encoder_features)
    
    y = Concatenate(axis=-1)([y, encoder_image_features])
    
    y = LayerNormalization(y.shape[2:])(y)
    
    #each LSTM unit returning a sequence of 6 outputs, one for each time step in the input data
    y = LSTM(units=12, dropout=0.2, return_sequences=True, activation='tanh')(y)
    y = LayerNormalization(y.shape[2:])(y)
    y = LSTM(units=20, dropout=0.2, return_sequences=True, activation='tanh')(y)
    y = LayerNormalization(y.shape[2:])(y)
    #output one time step from the sequence for each time step in the input but process 5 outputs of the input sequence at a time
    #y = TimeDistributed(Dense(units=5, activation='tanh'))(y)
    #attention_layer = attention()(y)
    y = LSTM(units=10, dropout=0.2, return_sequences=False, activation='tanh')(y)
    y = RepeatVector(21)(y)
    return y

def decoder(decoder_features, encoder_outputs, decoder_image_features):
    x = Concatenate(axis=-1)([decoder_features, encoder_outputs, decoder_image_features])
    # x = Add()([decoder_features, encoder_outputs]) 
    x = Masking(mask_value = -1000.)(x)
    x = TimeDistributed(Dense(units=20, activation='relu'))(x)
    x = TimeDistributed(Dense(units=8, activation='relu'))(x)
    y = TimeDistributed(Dense(units=1, activation='sigmoid'))(x)
    return y

def combine_model(X_encoder, X_decoder, img_encoder, img_decoder, resnet=True, train_resnet=False):
    
    # define input shapes
    encoder_features = Input(shape=X_encoder.shape[1:])
    decoder_features = Input(shape=X_decoder.shape[1:])
    encoder_img_features = Input(shape=img_encoder.shape[1:])
    decoder_img_features = Input(shape=img_decoder.shape[1:])
    
    #get cnn output for encoder/decoder
    encoder_img_output = image_model(encoder_img_features, resnet=resnet, train_resnet=train_resnet)
    decoder_img_output = image_model(decoder_img_features, resnet=resnet, train_resnet=train_resnet)
        
    #set encoder
    encoder_outputs = encoder(encoder_features, encoder_img_output)
    
    #set dedocer
    decoder_outputs = decoder(decoder_features, encoder_outputs, decoder_img_output)
    
    #set model
    model = Model([encoder_features, decoder_features, encoder_img_features, decoder_img_features], decoder_outputs)
    
    #print(Fore.YELLOW + f"\nCombined model..." + Style.RESET_ALL)
    
    return model

def compile_model(model):
    
    initial_learning_rate = 0.001

    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=1e-6)

    adam = Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=Precision())
    
    #print(Fore.YELLOW + f"\Compile model..." + Style.RESET_ALL)

    return model