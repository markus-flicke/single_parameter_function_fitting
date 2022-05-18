from sklearn.preprocessing import MinMaxScaler

def min_max_scale(x,y):
    scale_x = MinMaxScaler()
    x = scale_x.fit_transform(x)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)
    return x,y