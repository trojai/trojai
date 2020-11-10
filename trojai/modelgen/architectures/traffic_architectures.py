import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

"""
Describes various architectures for traffic sign detection,
"""


class GTSRBTrafficSignRecognition(nn.Module):
    """
    This the same NN architecture as the one specified by the Traffic Sign Detection based on
    Synthetically generated Traffic Signs.
    See here for more info:
    https://github.com/alexandrosstergiou/Traffic-Sign-Recognition-basd-on-Synthesised-Training-Data

    ***** The DNN architecture is as follows (Keras) *****

    def cnn_model(height,width,depth,number_signs):

        inp = Input(shape=(height, width, depth))

        conv_1 = Conv2D(32, (3, 3), padding='same')(inp)
        norm_1 = BatchNormalization()(conv_1)
        act1 = ELU(alpha=0.001)(norm_1)
        conv_2 = Conv2D(32, (3, 3), padding='same')(act1)
        norm_2 = BatchNormalization()(conv_2)
        act2 = ELU(alpha=0.001)(norm_2)
        pool_1 = MaxPooling2D((2, 2), data_format="channels_last")(act2)
        drop_1 = Dropout(0.2)(pool_1)

        conv_3 = Conv2D(64, (3, 3), padding='same')(drop_1)
        norm_3 = BatchNormalization()(conv_3)
        act3 = ELU(alpha=0.001)(norm_3)
        conv_4 = Conv2D(64, (3, 3), padding='same')(act3)
        norm_4 = BatchNormalization()(conv_4)
        act4 = ELU(alpha=0.001)(norm_4)
        pool_2 = MaxPooling2D((2, 2), data_format="channels_last")(act4)
        drop_2 = Dropout(0.2)(pool_2)

        conv_5 = Conv2D(128, (3, 3), padding='same')(drop_2)
        norm_5 = BatchNormalization()(conv_5)
        act5 = ELU(alpha=0.001)(norm_5)
        conv_6 = Conv2D(128, (3, 3), padding='same')(act5)
        norm_6 = BatchNormalization()(conv_6)
        act6 = ELU(alpha=0.001)(norm_6)
        pool_3 = MaxPooling2D((2, 2), data_format="channels_last")(act6)
        drop_3 = Dropout(0.2)(pool_3)


        flat = Flatten()(drop_3)
        hidden = Dense(512)(flat)
        norm_7 = BatchNormalization()(hidden)
        act7 = ELU(alpha=0.001)(norm_7)
        drop_4 = Dropout(0.5)(act7)
        out = Dense(number_signs, activation='softmax')(drop_4)

        model = Model(inputs=inp, outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = cnn_model(48,48,3,number_signs)
    ** Output **
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         (None, 48, 48, 3)         0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 48, 48, 32)        896
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 48, 48, 32)        128
    _________________________________________________________________
    elu_1 (ELU)                  (None, 48, 48, 32)        0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 48, 48, 32)        9248
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 48, 48, 32)        128
    _________________________________________________________________
    elu_2 (ELU)                  (None, 48, 48, 32)        0
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 24, 24, 32)        0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 24, 24, 64)        256
    _________________________________________________________________
    elu_3 (ELU)                  (None, 24, 24, 64)        0
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 24, 24, 64)        36928
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 24, 24, 64)        256
    _________________________________________________________________
    elu_4 (ELU)                  (None, 24, 24, 64)        0
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 12, 12, 64)        0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 12, 12, 128)       73856
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 12, 12, 128)       512
    _________________________________________________________________
    elu_5 (ELU)                  (None, 12, 12, 128)       0
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 12, 12, 128)       147584
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 12, 12, 128)       512
    _________________________________________________________________
    elu_6 (ELU)                  (None, 12, 12, 128)       0
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 128)         0
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 6, 6, 128)         0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4608)              0
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               2359808
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 512)               2048
    _________________________________________________________________
    elu_7 (ELU)                  (None, 512)               0
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 512)               0
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                25650
    =================================================================
    Total params: 2,676,306
    Trainable params: 2,674,386
    Non-trainable params: 1,920

    """
    def __init__(self, num_classes=5):
        super(GTSRBTrafficSignRecognition, self).__init__()
        elu_alpha = 0.001
        input_img_h = 128
        input_img_w = 128
        input_nchan = 3
        self.convnet = nn.Sequential(

            nn.Conv2d(input_nchan, 32, kernel_size=(3, 3), padding=1),  # [nbatch,3,128,128] -> [nbatch,32,128,128]
            nn.BatchNorm2d(32),
            nn.ELU(alpha=elu_alpha),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1), # [nbatch,32,128,128] -> [nbatch,32,128,128]
            nn.BatchNorm2d(32),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=2),  # TODO: do we need to specify the data channels format like in Keras?
                                          # [nbatch,32,128,128] -> [nbatch,32,64,64]
            nn.Dropout(p=0.2),            # what about Dropout2d

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # [nbatch,32,64,64] -> [nbatch,64,64,64]
            nn.BatchNorm2d(64),
            nn.ELU(alpha=elu_alpha),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),  # [nbatch,64,64,64] -> [nbatch,64,64,64]
            nn.BatchNorm2d(64),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=2),  # TODO: data channels format, do we need to specify that somehow?
                                          # [nbatch,64,64,64] -> [nbatch,64,32,32]
            nn.Dropout(p=0.2),                       # what about Dropout2d

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), # [nbatch,64,32,32] -> [nbatch,128,32,32]
            nn.BatchNorm2d(128),
            nn.ELU(alpha=elu_alpha),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), # [nbatch,128,32,32] -> [nbatch,128,32,32]
            nn.BatchNorm2d(128),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=2),  # TODO: data channels format, do we need to specify that somehow?
                                          # [nbatch,128,32,32] -> [nbatch,128,16,16]
            nn.Dropout(p=0.2),                       # what about Dropout2d
        )

        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 512),
            nn.BatchNorm1d(512),
            nn.ELU(alpha=elu_alpha),
            nn.Dropout(p=0.5),  # what about Dropout2d
            nn.Linear(512, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)       # equivalent of flatten in Keras
        output = self.fc(output)
        return output
