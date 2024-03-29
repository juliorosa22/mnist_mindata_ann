from keras_basic import *


def getIndexData(l_indexes):

    val_idx=[l_indexes[0][:],l_indexes[1][:]]
    train_idx=[l_indexes[i][:] for i in range(2,len(l_indexes)) ]

    val_idx = np.vstack(val_idx)
    train_idx = np.vstack(train_idx)

    train_idx = np.reshape(train_idx,(train_idx.shape[0]*train_idx.shape[1]))
    val_idx = np.reshape(val_idx,(val_idx.shape[0]*val_idx.shape[1]))
    val_idx=np.sort(val_idx).astype(int)
    train_idx=np.sort(train_idx).astype(int)

    lval_idx = val_idx.tolist()
    ltrain_idx=train_idx.tolist()
    return (ltrain_idx,lval_idx)


class dataGenerator(Sequence):

    def __init__(self,x,y,n_classes,list_index,batch_size=32):
        self.x=x
        self.y=y
        self.n_classes=n_classes
        self.batch_size=batch_size

        n_batches = int(np.ceil(len(list_index)/ float(self.batch_size)))

        copy_list = list_index[:]
        slice = len(copy_list) - (n_batches-1)*self.batch_size
        norm_list = copy_list[:-slice]
        rest_list=copy_list[-slice:]

        nlist_array = np.array(norm_list).astype(int)
        rest_array = np.array(rest_list).astype(int)
        final_list = np.split(nlist_array,(n_batches-1))
        final_list.append(rest_array)

        self.list_index=final_list


    def __len__(self):
        return len(self.list_index)

    def __getitem__(self, idx):

        x = self.x[(self.list_index[idx]).tolist()]
        y = self.y[(self.list_index[idx]).tolist()]

        y = to_categorical(y,num_classes=self.n_classes)

        return (x,y)
