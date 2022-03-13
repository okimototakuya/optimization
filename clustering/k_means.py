import numpy as np
import pandas as pd
from sklearn import datasets


class CentroidInitialValue():
    '''
    重心の初期値
    '''
    def __init__(self, cluster_num, observed_data):
        feature_dim = len(observed_data[0])     # 特徴量の次元
        ## 全データ対応
        self.__centroids_pre = [[np.random.uniform()*10 for _ in range(feature_dim)] for _ in range(cluster_num)] # 重心の初期値: ランダム入力
        self.__centroids_cur = [[np.random.uniform()*10 for _ in range(feature_dim)] for _ in range(cluster_num)]
        ## iris
        #self.__centroids_pre = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 重心の初期値: 手入力 (１つ前/現在)
        #self.__centroids_cur = [[5, 3, 1.5, 0.3], [6, 3, 5, 2], [7, 3, 6, 2]]
        #self.centroids_pre = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # 重心の初期値: 手入力 (１つ前/現在)
        #selfcentroids_cur = [[7, 5, 4, 0], [6, 3, 5, 2], [7, 3, 6, 2]]
        ## wine: curについて、原データのインデックス0, 63, 138の値を引用
        #self.__centroids_pre = [[0 for _ in range(13)] for _ in range(3)]  # 重心の初期値: 手入力 (１つ前/現在)
        #self.__centroids_cur = [[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00, 3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00, 1.065e+03],   \
        #                        [1.237e+01, 1.130e+00, 2.160e+00, 1.900e+01, 8.700e+01, 3.500e+00, 3.100e+00, 1.900e-01, 1.870e+00, 4.450e+00, 1.220e+00, 2.870e+00, 4.200e+02],    \
        #                        [1.349e+01, 3.590e+00, 2.190e+00, 1.950e+01, 8.800e+01, 1.620e+00, 4.800e-01, 5.800e-01, 8.800e-01, 5.700e+00, 8.100e-01, 1.820e+00, 5.800e+02]]

    @property
    def centroids_pre(self):
        pass

    @property
    def centroids_cur(self):
        pass

    @centroids_pre.getter
    def centroids_pre(self):
        return self.__centroids_pre

    @centroids_cur.getter
    def centroids_cur(self):
        return self.__centroids_cur


def k_means(cluster_num, observed_data, max_iter):
    '''
    k-means法

    Input
    -----
    - cluster_num: int型
        機械学習ユーザが、仮定するクラスタ数。
    - observed_data: numpy.ndarray型
        # 観測データ。クラスタリングの対象になる。
        # 行: 各観測データ
        # データ: 各々のデータの特徴量
    - max_iter: int型
        最大反復回数
    '''
    centroid_initial_value = CentroidInitialValue(cluster_num, observed_data)
    centroids_pre, centroids_cur =   \
        centroid_initial_value.centroids_pre, centroid_initial_value.centroids_cur                  # 重心の初期値
    which_cluster_pre, which_cluster_cur =  \
        np.zeros(len(observed_data), dtype='int64'), np.zeros(len(observed_data), dtype='int64')    # 各観測データの所属クラスタ
    iter_num = 0                    # 現在のステップ回数
    bool_iter_break = False         # 重心座標/クラスタ値が、収束したかどうか
    for iter_ in range(max_iter):
        if bool_iter_break == True:             # 重心座標/クラスタ値の更新がない場合、終了。
            break
        iter_num = iter_ + 1                    # 現在のステップ数
        which_cluster_pre = which_cluster_cur.copy()
        for i in range(len(observed_data)):
            div = [np.sqrt(sum(np.square(observed_data[i] - centroids_cur[j]))) for j in range(cluster_num)] # L2ノルム
            which_cluster_cur[i] = div.index(min(div))
        print('現在のクラスタ値')
        print('----------')
        print('{which_cluster_cur}'.format(which_cluster_cur=which_cluster_cur))
        for i_cluster in range(cluster_num):    # 重心の再計算
            index_i_cluster = [i for i, x in enumerate(which_cluster_cur) if x == i_cluster]        # クラスタiのインデックスを取得
            if len(index_i_cluster) != 0:   # 重心座標の更新計算について、0で割るわけにはいかないため。
                centroids_pre = centroids_cur.copy()
                # HACK: 2022.3.13: ↓重心座標の更新計算について、クラスタ０の座標だけnumpy.ndarray型で返されしまう。
                centroids_cur[i_cluster] = sum([observed_data[i] for i in index_i_cluster]) / len(index_i_cluster) # 各観測データ重み均等重心
                #if centroids_pre == centroids_cur:                     # 重心座標の更新がない場合、終了。
                if which_cluster_pre.all() == which_cluster_cur.all():  # クラスタ値の更新がない場合、終了。
                    bool_iter_break = True
                    break
        print('重心座標')
        print('----------')
        print('(１つ前, 現時点)=({pre}, {cur})'.format(pre=centroids_pre, cur=centroids_cur))
    return which_cluster_cur, iter_num


def main():
    '''
    k-means法により、クラスタリングを実行する。
    '''
    # データのロード
    #data_ = datasets.load_iris()        # iris: アヤメ
    #data_ = datasets.load_digits()      # digits: 手書きの数字
    data_ = datasets.load_wine()      # wine: ワイン
    df_data = pd.DataFrame(
        data = data_.data,               # 各観測データの値 (座標) : 各特徴量の値
        columns = data_.feature_names    # 各観測データのクラスタ値: 各特徴量の名称
        )
    cluster, iter_num = k_means(3, df_data.values, 2000)
    print('ステップ数: {iter_num}'.format(iter_num=iter_num))
    print('クラスタ値: {cluster}'.format(cluster=cluster[::]))
    print('正解率: {correct_rate}'.format(correct_rate= \
            sum([1 if val_cluster == data_.target[i] else 0 for i, val_cluster in enumerate(cluster)]) / len(data_.data)))

if __name__ == '__main__':
    main()
