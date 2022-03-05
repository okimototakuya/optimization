import sys
sys.path.append('../distribution/src/density-function/main/')
import gauss
import numpy as np
from matplotlib import pyplot as plt

N = 1000                        # 観測データの個数
#x_observed = np.random.randn(N) # 観測データ
x_observed = np.random.normal(5, 1, N) # 観測データ

def target_func(x):
    '''
    目的関数

    Notes
    -----
    - 関数A
        # 極値x=0
    - 関数B
        # 極値なし
        # 初期値, 学習率いずれを調整しても収束しない。
            → OverflowError
            → 極値がある場合と異なり、微分係数が0に収束しないことに由来する。
    - 関数C
        # 極値x=1, 3
    - 関数D
        # ガウシアンモデル(最小二乗法)を、勾配降下法で解く。
        # log(ab)=log(a)+log(b) → 二乗和の足し合わせ(sum)で表現できる。
        # 1/Nを掛けることで、二乗損失の算術平均になる。
    '''

    #y = x ** 2                     # 関数A
    #y = x ** 3                     # 関数B
    #y = (1/3)*x**3 - 2*x**2 + 3*x  # 関数C
    mu = x                          # 関数D
    y = (1/N) * sum([np.log(gauss.gauss(x_observed[i], mu, 1)) for i in range(N)])
    return y

def gradient(x):
    '''
    影響関数 (目的関数の微分)

    Notes
    -----
    - 2022.3.5: HACK: 式(D-2)が正しいように思えるが、上手くいかない。
    '''
    #y = 2 * x              # 関数A
    #y = 3 * x ** 2         # 関数B
    #y = x**2 - 4*x + 3     # 関数C
    mu = x                  # 関数D
    y = (1/N) * sum([(1/np.exp(((mu-x_observed[i])**2)/2) *  \
            (-(mu-x_observed[i])) * \
            np.exp(-(mu-x_observed[i])**2/2)) for i in range(N)])
    #y = -1 * sum([(mu-x_observed[i]) for i in range(N)])   ... (D-2)
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
    x_init = 4             # 初期値
    x_pred = [x_init if i == 0 else 0 for i in range(max_iteration)]    # 予測値のリスト
    for i in range(max_iteration-1):
        x_pred[i+1] = x_pred[i] + learning_rate * gradient(x_pred[i])
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
    plt.grid()
    plt.show()
    plt.close()
    # -------------------------------------------------------------------

if __name__ == '__main__':
    main()
