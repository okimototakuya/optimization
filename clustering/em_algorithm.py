import sys
sys.path.append('../../distribution/src/sampling')
sys.path.append('../../distribution/src/density-function/main')
import numpy as np
import sampling
import gauss


class observed():
    '''
    em_algorithmモジュールの、グローバル変数

    Notes
    -----
    - 観測データについての設定

    Usage
    -----
    - 本モジュールのグローバル変数設定
      ^ 本モジュールmain関数呼出によるもの
        → ユーザによる、本クラスインスタンス生成は不要
        → グローバルなスコープでインスタンスを生成し、値をグローバル変数に代入した。
      ^ 別モジュールからのグローバル変数設定
        → 本インスタンスを生成し、(必要に応じて)別モジュール内でグローバル化
    '''
    def __init__(self):
        # 観測点
        # -----
        # 真の分布(GMM。自作のsamplingモジュールで定義。)に基づいて、観測点を生成。
        # 2022.2.25: samplingモジュール内で、乱数のシードを固定できる仕様にしてもいいかも。
        sampling.sample_size = 100
        sampling.sample_mixed_gauss(mu=[0, 5], sigma=[1, 1], rate=[4/5, 1/5])   # 真の分布
        # グローバル変数
        # -----
        self.__x_observed = sampling.sample_list    # 観測データの値
        self.__n = len(self.__x_observed)           # 観測データの個数
        self.__d = 1                                # GMMの次元数

    @property
    def observed(self):
        '''
        ゲッター
        '''
        # 観測データの個数
        # GMMの次元数
        # 観測データの値
        return self.__x_observed, self.__n, self.__d


x_observed, n, d = observed().observed    # グローバル変数の設定

def phi(x, mu, sig):
    '''
    GMMについて、各峰のガウス分布の密度関数を返す。
    '''
    #return (1/(2*np.pi*sig**2)) * np.exp(-(x-mu)**2/(2*sig**2))
    return gauss.gauss(x, mu, sig)

#def target_func(x_observed, *theta):
#    '''
#    目的関数
#
#    Notes
#    -----
#    - 応用的に最適化の対象となる関数(目的関数)
#      ^ 例1. 混合ガウスモデルの対数尤度
#      ^ 例2. ↑より一般的に、不完全なデータ
#    '''
#    w, mu, sig = theta[0], theta[1], theta[2]
#    m = len(w)  # 多峰分布の峰の個数
#    n = len(x_observed) # 観測データの個数
#    # GMMの対数尤度
#    # -----
#    # ↓損失について、平均をとらずに単に合計している。
#    log_likelihood = sum([np.log(sum([w[l]*phi(x_observed[i], mu[l], sig[l]) for l in range(m)]))   \
#                            for i in range(n)])
#    return log_likelihood

def e_step(*theta):
    '''
    EMアルゴリズムのEステップ

    Note
    -----
    - 現ステップの解から、媒介変数を計算する。
    - 現ステップの解θ_i (θ^) を通る目的関数(例.対数尤度)の下界b(θ)を求める。

    Input
    -----
    - *theta: list
        現ステップの解 (w, mu, sig)

    Return
    -----
    - eta: list
        媒介変数
    '''
    w, mu, sig = theta[0], theta[1], theta[2]
    global n, x_observed
    m = len(w)                                          # 多峰分布の峰の個数
    eta = [[0 for _ in range(m)] for _ in range(n)]     # 媒介変数
    x = x_observed                                      # 観測データの値
    for i in range(n):
        for l in range(m):
            eta[i][l] = (w[l]*phi(x[i], mu[l], sig[l]**2)) / \
                    sum([w[l_]*phi(x[i], mu[l_], sig[l_]**2) for l_ in range(m)])
    return eta

def m_step(eta):
    '''
    EMアルゴリズムのMステップ

    Note
    -----
    - 現ステップまでの媒介変数の集合から、解を計算する。
    - 目的関数(例.対数尤度)の下界b(x)を最大にするパラメータxを求める。

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
    global n, d, x_observed
    x = x_observed                  # 観測データの値
    m = len(eta[0])                 # 多峰分布の峰の個数
    w  = [0 for _ in range(m)]      # 重みパラメータ
    mu = [0 for _ in range(m)]      # 期待値
    sig = [0 for _ in range(m)]     # 標準偏差
    for l in range(m):
        w[l] = (1/n) * sum([eta[i][l] for i in range(n)])
        mu[l] = sum([eta[i][l]*x[i] for i in range(n)]) /   \
                sum([eta[i_][l] for i_ in range(n)])
        sig[l] = np.sqrt(sum([eta[i][l]*np.square(x[i]-mu[l]) for i in range(n)]) /   \
                (d*sum(eta[i_][l] for i_ in range(n)))) # FIXME: 2022.2.24: 変数dの正体が不明
    return w, mu, sig

def main():
    '''
    EMアルゴリズムの実行

    Notes
    -----
    - 応用的に最適化の対象となる関数(目的関数)
      ^ 例1. 混合ガウスモデルの対数尤度
      ^^ 応用例. 確率分布推定に基づくクラスタリング
      ^^ 関連1. 階層的クラスタリング (クラスター分析)
      ^^ 関連2. 非階層的クラスタリング (k平均法 (k-means clustering))
      ^^ https://datachemeng.com/gaussianmixturemodel/
      ^ 例2. ↑より一般的に、不完全なデータ
    '''
    # 初期値
    # 2022.2.25注
    # -----
    # GMMの対数尤度の最適化では、パラメータ(最適化の各ステップで更新される変数)は
    # θ=(w_1,...,w_n, mu_1^T,...,mu_n^T, sig_1,...,sig_n)
    #x = [[-5, 5]]      # GMMのパラメータの推定という目的では、用いない。
    w = [7/10, 3/10]    # GMMのパラメータ: w, mu, sig
    mu = [-1, 6]
    sig = [1, 1]
    # 反復回数
    n_iter = 1000
    for i in range(n_iter): # FIXME: 2022.2.24: xの更新は、いつされるのか?
        eta = e_step(w, mu, sig)
        w, mu, sig = m_step(eta)
        print('{i}ステップ目: [w, mu, sig]=[{w}, {mu}, {sig}]'.format(i=i, w=w, mu=mu, sig=sig))

if __name__ == '__main__':
    main()
