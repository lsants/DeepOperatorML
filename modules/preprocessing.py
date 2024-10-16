import numpy as np

def preprocessing(u, g_u, train_perc=0.8,
                  train_idx=None, val_idx=None, test_idx=None):
    
    if (train_idx is not None) and (test_idx is not None):
        u_train, g_u_train = u[train_idx[0]:train_idx[1]], g_u[train_idx[0]:train_idx[1]]
        u_val, g_u_val = u[val_idx[0]:val_idx[1]], g_u[val_idx[0]:val_idx[1]]
        u_train, g_u_train = np.concatenate([u_train, u_val], axis=0), np.concatenate([g_u_train, g_u_val], axis=0)
        u_test, g_u_test = u[test_idx[0]:test_idx[1]], g_u[test_idx[0]:test_idx[1]]
    else:        
        u_train, g_u_train, u_test, g_u_test, = train_test_split(u=u,
                                                                g_u=g_u,
                                                                train_perc=train_perc)
    g_u_real_train, g_u_imag_train = extract_real_imag_parts(g_u_train)
    g_u_real_test, g_u_imag_test = extract_real_imag_parts(g_u_test)

    return dict({
    'u_train': u_train,
    'u_test': u_test,
    'g_u_real_train': g_u_real_train,
    'g_u_real_test': g_u_real_test,
    'g_u_imag_train': g_u_imag_train,
    'g_u_imag_test': g_u_imag_test,
    })

def train_test_split(u, g_u, xt=None, train_perc=0.8):
    """
    Splits u, xt and g_u into training set.

    Params:
        @ batch_xt: trunk in batches

    if batch_xt:
        @ u.shape = [bs, x_len]
        @ xt.shape = [bs, x_len*t_len, 3]
        @ g_u.shape = [bs, x_len*t_len] 
    else:
        @ u.shape = [bs, x_len]
        @ xt.shape = [x_len*t_len, 2]
        @ g_u.shape = [bs, x_len*t_len] 
    """
    
    def _split(f, train_size):
        """
        Splits f into train and test sets.
        """
        if isinstance(f, (list, tuple)):
            train, test = list(), list()
            for i, f_i in enumerate(f):
                train.append(f_i[:train_size])
                test.append(f_i[train_size:])
                assert(train[i].shape[-1]==test[i].shape[-1])
        else:            
            train, test = f[:train_size], f[train_size:]
            assert(train.shape[-1]==test.shape[-1])

        return train, test

    if train_perc > 0.0:
        train_size = int(np.floor(int(u.shape[0])*train_perc))

        u_train, u_test = _split(u, train_size)
        g_u_train, g_u_test = _split(g_u, train_size)

        return u_train, g_u_train, u_test, g_u_test
    
    else:
        return None, None, u, g_u, xt, xt
    
def extract_real_imag_parts(arr: np.ndarray[np.complex128]):
    real_part = arr.real
    imaginary_part = arr.imag
    return real_part, imaginary_part