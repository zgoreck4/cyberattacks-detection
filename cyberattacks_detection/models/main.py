from .RBFNN import RBFNN
from .ELM import ELM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

def plot_data(x, y, title):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"img\{title}.png", 
            bbox_inches ="tight") 
    plt.show()

def plot_data_2dim(x, y, title):
    plt.scatter(x[:, 0], y[:, 0], label='wyjście y1')
    plt.scatter(x[:, 1], y[:, 1], label='wyjście y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.savefig(f"img\{title}_2dim.png", 
            bbox_inches ="tight") 
    plt.show()

def plot_real_pred_2dim(x, y, y_pred, title, nn):
    plt.scatter(x[:, 0], y[:, 0], label='Rzeczywiste y1')
    plt.scatter(x[:, 1], y[:, 1], label='Rzeczywiste y2')
    plt.scatter(x[:, 0], y_pred[:, 0], label='Przewidywane y1')
    plt.scatter(x[:, 1], y_pred[:, 1], label='Przewidywane y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.savefig(f"img\{nn}_{title}_2dim.png", 
            bbox_inches ="tight") 
    plt.show()

def plot_y_real_pred(y, y_pred, title, nn):
    plt.scatter(y[:, 0], y_pred[:, 0], label='Wyjście y1')
    plt.scatter(y[:, 1], y_pred[:, 1], label='Wyjście y2')
    y_good = np.linspace(min(y[:, 0]), max(y[:, 0]), num=50)   
    plt.plot(y_good, y_good, 'r--')
    plt.xlabel('Rzeczywiste y')
    plt.ylabel('Przewidywane y')
    plt.legend()
    plt.title(title)
    plt.savefig(f"img\{nn}_{title}_multidim.png", 
            bbox_inches ="tight") 
    plt.show()

def plot_real_pred(x, y, y_pred, title, nn):
    plt.scatter(x, y, label='Rzeczywiste')
    plt.scatter(x, y_pred, label='Przewidywane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.savefig(f"img\{nn}_{title}.png", 
            bbox_inches ="tight") 
    plt.show()

def main():
    # wybór sieci neuronowej
    # nn = 'RBF'
    nn = 'ELM'

    # Example usage 1 input - 1 output
    x = np.linspace(-5, 5, num=200).reshape((-1, 1))
    y = 2.0 + 0.5 * x + 1.5 * x**2 + x**3 + np.random.normal(scale=3.0, size=x.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Visualize dataset
    plot_data(X_train, y_train, "Zbiór uczący")
    plot_data(X_test, y_test, "Zbiór testowy")

    if nn == 'RBF':

        n_centers = 4

        rbf_net = RBFNN(n_centers, alpha=0.001)
        rbf_net.fit(X_train, y_train, np.min(X_train, axis=0), np.max(X_train, axis=0), np.min(y_train, axis=0), np.max(y_train, axis=0))
        y_train_pred = rbf_net.predict(X_train)
        y_test_pred = rbf_net.predict(X_test)
    
    elif nn == 'ELM':
        elm_net = ELM(X_train.shape[1], 7)
        elm_net.fit(X_train, y_train, np.min(X_train, axis=0), np.max(X_train, axis=0), np.min(y_train, axis=0), np.max(y_train, axis=0))
        y_train_pred = elm_net.predict(X_train)
        y_test_pred = elm_net.predict(X_test)
    
    print("dla pierwszego zbioru uczącego MSE: %2.3f" % (MSE(y_train, y_train_pred)))
    print("dla pierwszego zbioru testowego MSE: %2.3f" % (MSE(y_test, y_test_pred)))

    plot_real_pred(X_train, y_train, y_train_pred, "Wyniki dla zbioru uczącego", nn)
    plot_real_pred(X_test, y_test, y_test_pred, "Wyniki dla zbioru testowego", nn)

    
    # Example usage 2 inputs - 2 outputs
    x1 = np.linspace(-5, 5, num=200).reshape((-1, 1))    
    x2 = np.linspace(-1, 4, num=200).reshape((-1, 1))
    y1 = 2.0 + 0.5 * x1 + 1.5 * x2**2 + x1**3 + np.random.normal(scale=3.0, size=x1.shape)
    y2 = 2.0 + 0.5 * x2 + 1.5 * x1**2 + x2**3 + np.random.normal(scale=3.0, size=x2.shape)
    x = np.concatenate((x1, x2), axis=1)
    y = np.concatenate((y1, y2), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Visualize 
    plot_data_2dim(X_train, y_train, "Zbiór uczący")
    plot_data_2dim(X_test, y_test, "Zbiór testowy")

    if nn == 'RBF':

        n_centers = 10

        rbf_net = RBFNN(n_centers, alpha=0.001)
        rbf_net.fit(X_train, y_train, np.min(X_train, axis=0), np.max(X_train, axis=0), np.min(y_train, axis=0), np.max(y_train, axis=0))
        y_train_pred = rbf_net.predict(X_train)
        y_test_pred = rbf_net.predict(X_test)

    elif nn == 'ELM':
        elm_net = ELM(X_train.shape[1], 10)
        elm_net.fit(X_train, y_train, np.min(X_train, axis=0), np.max(X_train, axis=0), np.min(y_train, axis=0), np.max(y_train, axis=0))
        y_train_pred = elm_net.predict(X_train)
        y_test_pred = elm_net.predict(X_test)

    print("dla drugiego zbioru uczącego MSE: %2.3f" % (MSE(y_train, y_train_pred)))
    print("dla drugiego zbioru testowego MSE: %2.3f" % (MSE(y_test, y_test_pred)))

    plot_y_real_pred(y_train, y_train_pred, "Wyniki dla zbioru uczącego", nn)   
    plot_y_real_pred(y_test, y_test_pred, "Wyniki dla zbioru testowego", nn)    


if __name__ == '__main__':
    main()