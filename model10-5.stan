// model10-5.stan
// モデル：規定【ローカルレベルモデル+周期モデル(時間領域アプローチ）、カルマンフィルタを活用】

data{
  int<lower=1>    t_max;           // 時系列長
  matrix[1, t_max]    y;           // 観測値

  matrix[12, 12]      G;           // 状態遷移行列
  matrix[12,  1]      F;           // 観測行列
  vector[12]         m0;           // 事前分布の平均ベクトル
  cov_matrix[12]     C0;           // 事前分布の共分散行列
}

parameters{
  real<lower=0>       W_mu;        // 状態雑音（レベル成分）の分散
  real<lower=0>       W_gamma;     // 状態雑音（　周期成分）の分散
  cov_matrix[1]       V;           // 観測雑音の共分散行列
}

transformed parameters{
    matrix[12, 12]    W;           // 状態雑音の共分散行列

    for (k in 1:12){               // Stanのmatrixでは列優先のアクセスが高速
      for (j in 1:12){
             if (j == 1 && k == 1){ W[j, k] = W_mu;     }
        else if (j == 2 && k == 2){ W[j, k] = W_gamma;  }
        else{                       W[j, k] = 0;        }
      }
    }
}

model{
  // 尤度の部分
  /* 線形・ガウス型状態空間モデルの尤度を求める関数 */
  y ~ gaussian_dlm_obs(F, G, V, W, m0, C0);

  // 事前分布の部分
  /* W, Vの事前分布：無情報事前分布（省略時のデフォルト設定を活用） */
}
