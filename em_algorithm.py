import numpy as np

def l_func(x, w, mu, sig):
    '''
    尤度関数

    Note
    -----
    - 仮定した分布
    - 最適化対象の対数尤度

    Input
    -----
    x: numpy.ndarray
        尤度関数の引数 (確率変数の値)

    Return
    -----
    phi: numpy.ndarray
        尤度関数の値 (確率値)
    '''
    m = len(w)  # 多峰分布の峰の個数
    phi = [w[i]*np.exp((x-mu[i])**2/(2*sig[i]**2)) for i in range(m)]
    return phi

def e_step(x, *solution):
    '''
    EMアルゴリズムのEステップ

    Note
    -----
    - 現ステップの解から、媒介変数を計算する。

    Input
    -----
    - x: int
        現ステップの値
    - *solution: list
        現ステップの解 (w, mu, sig)

    Return
    -----
    - eta: list
        媒介変数
    '''
    w, mu, sig = solution[0], solution[1], solution[2]
    m = len(w)                  # 多峰分布の峰の個数
    eta = [0 for _ in range(m)] # 媒介変数
    for l in range(m):  # 分布の峰の数だけ繰り返し
        eta[l] = w[l]*l_func(x, mu[l], sig[l]) / \
                sum([w[l_]*l_func(x, mu[l_], sig[l_]) for l_ in range(m)])
    return eta

def m_step(x, eta):
    '''
    EMアルゴリズムのMステップ

    Note
    -----
    - 現ステップまでの媒介変数の集合から、解を計算する。

    Input
    -----
    - x: list
        現ステップまでの値の集合
    - eta: list
        現ステップまでの媒介変数の集合

    Return
    -----
    - w, mu, sig: tuple
        解
    '''
    m = len(eta[0])                 # 多峰分布の峰の個数
    n = len(eta)                    # これまでのステップ数の合計
    w  = [0 for _ in range(m)]      # 重みパラメータ
    mu = [0 for _ in range(m)]      # 期待値
    sig = [0 for _ in range(m)]     # 標準偏差
    for l in range(m):
        w[l] = (1/n) * sum(eta[i][l] for i in range(n))
        mu[l] = sum([eta[i][l]*x[i] for i in range(n)]) /   \
                sum([eta[i][l] for i in range(n)])
        sig[l] = np.sqrt(sum([eta[i][l]*np.square(x[i]-mu[l]) for i in range(n)]) /   \
                d*sum(eta[i][l] for i in range(n))) # FIXME: 2022.2.24: 変数dの正体が不明
    return w, mu, sig

def main():
    '''
    EMアルゴリズムの実行
    '''
    # 初期値
    x = [[-5, 5]]
    w = [9/10, 1/10]
    mu = [-5, 5]
    sig = [1, 1]
    # 反復回数
    n_iter = 1000
    for i in range(n_iter): # FIXME: 2022.2.24: xの更新は、いつされるのか?
        eta = e_step(x, w, mu, sig)
        w, mu, sig = m_step(x, eta)

if __name__ == '__main__':
    main())
