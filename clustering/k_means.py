import numpy as np
import pandas as pd
from sklearn import datasets

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
    feature_dim = len(observed_data[0])                                                                 # 特徴量の次元
    #centroids = [[np.random.uniform()*10 for _ in range(feature_dim)] for _ in range(cluster_num)]     # 重心の初期値: ランダム入力
    centroids = [[5, 3, 1.5, 0.3], [6, 3, 5, 2], [7, 3, 6, 2]]                                          # 重心の初期値: 手入力
    which_cluster = np.zeros(len(observed_data), dtype='int64')                                         # 各観測データの所属クラスタ
    for _ in range(max_iter):
        for i in range(len(observed_data)):
            div = [np.sqrt(sum(np.square(observed_data[i] - centroids[j]))) for j in range(cluster_num)] # L2ノルム
            which_cluster[i] = div.index(min(div))
        for i_cluster in range(cluster_num):    # 重心の再計算
            index_i_cluster = [i for i, x in enumerate(which_cluster) if x == i_cluster]                        # クラスタiのインデックスを取得
            if len(index_i_cluster) != 0:
                centroids[i_cluster] = sum([observed_data[i] for i in index_i_cluster]) / len(index_i_cluster) # 各観測データ重み均等重心
        # 2022.3.3[HACK]: もしも重心が変わっていなければ、終了
    return which_cluster


def main():
    '''
    k-means法により、クラスタリングを実行する。
    '''
    # データのロード
    iris = datasets.load_iris()
    df_iris = pd.DataFrame(
        data = iris.data,               # 各観測データの値 (座標) : 各特徴量の値
        columns = iris.feature_names    # 各観測データのクラスタ値: 各特徴量の名称
        )
    cluster = k_means(3, df_iris.values, 2000)
    print('クラスタ値: {cluster}'.format(cluster=cluster[::]))
    print('正解率: {correct_rate}'.format(correct_rate= \
            sum([1 if val_cluster == iris.target[i] else 0 for i, val_cluster in enumerate(cluster)]) / len(iris.data)))

if __name__ == '__main__':
    main()
