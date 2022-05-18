def plot_model(x, y, y_pred, model_name):
    import matplotlib.pyplot as plt
    plt.scatter(x,y, label='Actual')
    plt.scatter(x,y_pred, label='Predicted')
    plt.title(model_name)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    from sklearn.metrics import mean_squared_error
    print(f'MSE: {mean_squared_error(y_pred, y):.2f}')
    plt.show()