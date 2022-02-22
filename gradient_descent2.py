import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_func(x, y):
    '''
    目的関数

    Return
    -----
    - z: int
        目的関数値
    '''
    z = x**2 + y**2     # 関数A
    return z

def gradient(x, y):
    '''
    勾配 (目的関数の微分)

    Return
    -----
    - dz: numpy.ndarray
        勾配ベクトル
    '''
    dzdx = 2*x                      # 関数A
    dzdy = 2*y
    dz = np.array([dzdx, dzdy])     ## 関数Aの勾配
    return dz

def main():
    '''
    勾配降下法の実行

    Notes
    -----
    - 2変量
    '''
    learning_rate = 0.1     # 学習率
    x_init = 10             # xの初期値
    y_init = 10             # yの初期値
    max_iteration = 1000    # 最大反復回数
    x_pred = [x_init if i == 0 else 0 for i in range(max_iteration)]
    y_pred = [y_init if i == 0 else 0 for i in range(max_iteration)]
    for i in range(max_iteration-1):
        x_pred[i+1], y_pred[i+1] = np.array([x_pred[i], y_pred[i]]) - learning_rate * gradient(x_pred[i], y_pred[i])
        print(i, x_pred[i])

    x_pred = np.array(x_pred)           # 描画用にx0をnumpy配列変換
    y_pred = np.array(y_pred)           # 描画用にx0をnumpy配列変換
    z_pred = target_func(x_pred, y_pred)          # 軌跡のz値を計算

    # 基準関数の表示用
    x = np.arange(-10, 11, 2)
    y = np.arange(-10, 11, 2)
    X, Y = np.meshgrid(x, y)
    Z = target_func(X, Y)

    # ここからグラフ描画----------------------------------------------------------------
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    #  グラフの入れ物を用意する。
    fig = plt.figure()
    ax1 = Axes3D(fig)

    # 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # データプロットする。
    ax1.plot_wireframe(X, Y, Z, label='f(x, y)')
    #ax1.scatter3D(x_pred, y_pred, z_pred, label='gd', color='red', s=50)
    ax1.scatter3D(x_pred, y_pred, z_pred, label='gd', color='red', s=50)

    # グラフを表示する。
    plt.show()
    plt.close()
    # ---------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
