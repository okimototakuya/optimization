import numpy as np
from matplotlib import pyplot as plt

def target_func(x):
    '''
    目的関数
    '''
    #y = x ** 2  # 関数A
    #y = x ** 3  # 関数B
    y = (1/3)*x**3 - 2*x**2 + 3*x # 関数C
    return y

def influence_func(x):
    '''
    影響関数 (目的関数の微分)
    '''
    #y = 2 * x   # 関数A
    #y = 3 * x ** 2  # 関数B
    y = x**2 - 4*x + 3  # 関数C
    return y

def main():
    '''
    勾配降下法による学習と、学習結果のプロットを行う。

    Notes
    ----
    - 実行結果
    　^ 局所解に収束する。
    　^ 収束せずに、オーバーフローを起こす。
    '''
    learning_rate = 0.1     # 学習率
    max_iteration = 1000    # 最大反復回数
    x_init = -2             # 初期値
    x_pred = [x_init if i == 0 else 0 for i in range(max_iteration)]    # 予測値のリスト
    for i in range(max_iteration-1):
        x_pred[i+1] = x_pred[i] + learning_rate * influence_func(x_pred[i])
        print(i, x_pred[i])
    x_pred = np.array(x_pred)         # numpy.ndarray型配列にキャスト

    # ここからグラフ描画-------------------------------------------------
    # 目的関数の表示用
    x = np.arange(-3, 6, 1)
    y = target_func(x)

    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')

    # 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # データプロットの準備。
    ax1.plot(x, y, lw=1)
    ax1.scatter(x_pred, target_func(x_pred), color='red')

    # グラフを表示する。
    fig.tight_layout()
    plt.show()
    plt.close()
    # -------------------------------------------------------------------

if __name__ == '__main__':
    main()
