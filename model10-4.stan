// model10-4.stan
// モデル：規定【ローカルレベルモデル+周期モデル(時間領域アプローチ）】

data{
  int<lower=1>    t_max;            // 時系列長
  vector[t_max]       y;            // 観測値

  vector[12]         m0;            // 事前分布の平均ベクトル
  cov_matrix[12]     C0;            // 事前分布の共分散行列
}

parameters{
  real              x0_mu;          // 状態（レベル成分）[0]
  vector[11]        x0_gamma;       // 状態（　周期成分）[0]
  vector[t_max]      x_mu;          // 状態（レベル成分）[1:t_max]
  vector[t_max]      x_gamma;       // 状態（　周期成分）[1:t_max]

  real<lower=0>      W_mu;          // 状態雑音（レベル成分）の分散
  real<lower=0>      W_gamma;       // 状態雑音（　周期成分）の分散
  cov_matrix[1]      V;             // 観測雑音の共分散行列
}

model{
  // 尤度の部分
  /* 観測方程式 */
  for (t in 1:t_max){
    y[t] ~ normal(x_mu[t] + x_gamma[t], sqrt(V[1, 1]));
  }

  // 事前分布の部分
  /* 状態（レベル成分）の事前分布 */
  x0_mu ~ normal(m0[1], sqrt(C0[1, 1]));

  /* 状態方程式（レベル成分） */
    x_mu[1] ~ normal(x0_mu     , sqrt(W_mu));
  for(t in 2:t_max){
    x_mu[t] ~ normal( x_mu[t-1], sqrt(W_mu));
  }

  /* 状態（周期成分）の事前分布 */
  for (p in 1:11){
    x0_gamma[p] ~ normal(m0[p+1], sqrt(C0[(p+1), (p+1)]));
  }

  /* 状態方程式（周期成分） */
    x_gamma[1] ~ normal(-sum(x0_gamma[1:11])                           ,
                                                                     sqrt(W_gamma));
  for(t in 2:11){
    x_gamma[t] ~ normal(-sum(x0_gamma[t:11])-sum(x_gamma[     1:(t-1)]),
                                                                     sqrt(W_gamma));
  }
  for(t in 12:t_max){
    x_gamma[t] ~ normal(                    -sum(x_gamma[(t-11):(t-1)]),
                                                                     sqrt(W_gamma));
  }

  /* W, Vの事前分布：無情報事前分布（省略時のデフォルト設定を活用） */
}
