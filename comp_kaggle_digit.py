from keras_basic import *
from dataGenerator import *


#dir = '/home/julio/min_dados/'
def trainDenseModel(model_name,train_gen,val_gen,n_epochs):
    print('Training model : ',model_name)
    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_acc',patience=50,verbose=1)

    inpL = Input(shape=(28*28,))
    hiddL = Dense(64,use_bias=True,activation='relu')(inpL)
    hiddL = Dropout(rate=0.2)(hiddL)
    hiddL = Dense(48,use_bias=True,activation='relu')(hiddL)
    hiddL = Dropout(rate=0.2)(hiddL)


    outL = Dense(train_gen.n_classes,use_bias=True,activation='softmax')(hiddL)
    model = Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())

    hist=model.fit_generator(generator=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[es])
    saveHistLog(model_name+'.h5',getArrayHistLog(hist))
    print("Treinamento completo")
    model.save(model_name+'.h5')
    return model


def trainConvModel(model_name,train_gen,val_gen,n_epochs):
    print('Training model : ',model_name)
    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=25,verbose=1)


    inpL = Input(shape=(28*28,))
    hiddL = Reshape((28,28,1))(inpL)
    hiddL = Conv2D(64, kernel_size=(4,4), strides = 1, activation='relu')(hiddL)
    hiddL = Dropout(0.2)(hiddL)
    hiddL = Conv2D(32, kernel_size=(4,4), strides = 1, activation='relu')(hiddL)
    hiddL = Dropout(0.2)(hiddL)
    hiddL = GlobalMaxPooling2D()(hiddL)
    hiddL=Dense(16,activation='relu')(hiddL)
    outL = Dense(10,activation='softmax')(hiddL)

    model = Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())

    hist=model.fit_generator(generator=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[es])
    saveHistLog(model_name+'.h5',getArrayHistLog(hist))
    print("Treinamento completo")
    model.save(model_name+'.h5')
    return model


def predictNewExamples(dset_name,model_ver):
    fh5 = h5.File(dset_name,'r')
    data=fh5['minist_dset'][:]
    print(data.shape)

    X=data[:]



    model_name = str(7)+'_'+model_ver
    model=load_model(model_name+'.h5')
    Y=model.predict(X)

    label = np.argmax(Y,1)
    lines = [['ImageId','Label']]
    print(label[:10])

    for i in range(label.shape[0]):
        lines.append([i+1,label[i]])


    with open('predicts.csv', 'w') as f:
        writer = csv.writer(f,dialect='excel')
        for row in lines:
            writer.writerow(row)
    f.close()


def test_model(dset_name,model_ver):
    fh5 = h5.File(dset_name,'r')
    data=fh5['minist_dset'][:]
    print(data.shape)
    Y = data[:,0]
    X=data[:,1:]


    n_classes=10
    indexes=np.arange(X.shape[0])
    print('Dataset size X:(%d,%d)| Y:(%d,)'%(X.shape[0],X.shape[1],Y.shape[0]))

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc = []
    loss_folds=[]
    

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver

        xtest = X[ix_folds[k]]
        ytest=to_categorical(Y[ix_folds[k]],num_classes=n_classes)
        model=load_model(model_name+'.h5')
        #print(model.summary())
        #print(xtest.shape,ytest.shape)
        info=model.evaluate(xtest,ytest)
        acc.append(info[1])
        print(info)
    acc=np.array(acc)
    print(acc.mean())
    print(np.std(acc))
    f = h5.File('info_folds.h5','w')
    dt=f.create_dataset('infos',acc.shape,dtype=np.float32)
    dt[:]=acc[:]
    f.close()

def buildModel(dset_name,model_ver):

    fh5 = h5.File(dset_name,'r')
    data=fh5['minist_dset'][:]
    print(data.shape)
    Y = data[:,0]
    X=data[:,1:]


    n_classes=10
    indexes=np.arange(X.shape[0])
    print('Dataset size X:(%d,%d)| Y:(%d,)'%(X.shape[0],X.shape[1],Y.shape[0]))

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]
    models_number=[6,9]
    for k in models_number:
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver

        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]
        i_train,i_val=getIndexData(idx_train)

        train_generator = dataGenerator(X,Y,n_classes,i_train,batch_size=128)
        val_generator = dataGenerator(X,Y,n_classes,i_val,batch_size=128)

        #model=trainDenseModel(model_name,train_generator,val_generator,300)
        model=trainConvModel(model_name,train_generator,val_generator,70)


buildModel('train.h5','dense_model_mnist_v2')
#m=load_model('8_conv_model.h5')
#plot_model(m,show_shapes=True,to_file='min_conv.png')
#test_model('train.h5','conv_model')
#predictNewExamples('test.h5','conv_model')
'''
#melhor configuracao atual
hiddL = Conv2D(64, kernel_size=(4,4), strides = 1, activation='relu')(hiddL)
#hiddL = MaxPooling2D(pool_size=(6,6),strides=1)(hiddL)
hiddL = Dropout(0.2)(hiddL)
hiddL = Conv2D(32, kernel_size=(4,4), strides = 1, activation='relu')(hiddL)
#hiddL = MaxPooling2D(pool_size=(6,6),strides=1)(hiddL)
hiddL = Dropout(0.2)(hiddL)
hiddL = GlobalMaxPooling2D()(hiddL)
#hiddL = Flatten()(hiddL)

hiddL=Dense(16,activation='relu')(hiddL)

'''

'''

FOlds p arrumar:

Média = 0.9696047
Desvio padrao nos folds= 0.004892321
'''
