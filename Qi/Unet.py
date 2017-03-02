from keras.models import Model;
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dense,Flatten, Deconvolution2D,Dropout,SpatialDropout2D;
from keras.layers.normalization import BatchNormalization as BN;
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras import backend as K
from keras.layers import Merge

def get_unet(img_rows, img_cols, version):
    if K.image_dim_ordering() == 'tf':
        c = 3;
        inputs = Input((img_rows, img_cols, 1))
    else:
        c = 1;
        inputs = Input((1,img_rows, img_cols))

    def unet_conv(x_in, nf, rep=1):
        x_out = BN(axis=c)(Convolution2D(nf, 3, 3, activation='relu', border_mode='same')(x_in));
        #x_out = LeakyReLU(0.1)(x_out);
        if rep>1:
            for i in range(rep-1):
                x_out = BN(axis=c)(Convolution2D(nf, 3, 3, activation='relu', border_mode='same')(x_out));
                #x_out = LeakyReLU(0.1)(x_out);
        return x_out;
    if version==1:
        n0 = 8;
        net = unet_conv(inputs, n0, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(net);
        net = unet_conv(net, n0*2, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(net);
        net = unet_conv(net, n0*4, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(net);
        net = unet_conv(net, n0*8, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(net);
        net = unet_conv(net, n0*16, rep=2);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*8, rep=2)
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*4, rep=2);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*2, rep=2);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0, rep=1);
        labels = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='labels')(net)
    elif version==2:
        n0 = 8;
        net = unet_conv(inputs, n0, rep=3);
        net = MaxPooling2D(pool_size=(2,2))(net);
        cov2 = unet_conv(net, n0*2, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov2);
        cov3 = unet_conv(net, n0*4, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov3);
        cov4 = unet_conv(net, n0*8, rep=2);
        net = UpSampling2D(size=(2,2))(cov4);
        net = unet_conv(net, n0*4, rep=2);
        net = unet_conv(merge([net,cov3],mode='concat',concat_axis=c), n0*4, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*2, rep=2);
        net = unet_conv(merge([net,cov2],mode='concat',concat_axis=c), n0*2, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0, rep=1);
        labels = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='labels')(net)
    elif version==3:
        n0 = 8;
        cov1 = unet_conv(inputs, n0, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov1);
        cov2 = unet_conv(net, n0*2, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov2);
        cov3 = unet_conv(net, n0*4, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov3);
        cov4 = unet_conv(net, n0*8, rep=2);
        net = MaxPooling2D(pool_size=(2,2))(cov4);
        cov5 = unet_conv(net, n0*16, rep=2);
        #5
        net = UpSampling2D(size=(2,2))(cov5);
        net = unet_conv(net, n0*8, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*4, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*2, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net5 = unet_conv(net, n0, rep=1);
        #4
        net = UpSampling2D(size=(2,2))(cov4);
        net = unet_conv(net, n0*4, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net = unet_conv(net, n0*2, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net4 = unet_conv(net, n0, rep=1);
        #3
        net = UpSampling2D(size=(2,2))(cov3);
        net = unet_conv(net, n0*2, rep=1);
        net = UpSampling2D(size=(2,2))(net);
        net3 = unet_conv(net, n0, rep=1);

        net = merge([net5,net4,net3],mode='concat',concat_axis=c);
        labels = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same', name='labels')(net)

    else:
        raise Exception("not defined net version")

    model = Model(input=inputs, output=labels)
    return model
