// model10-1.stan
// モデル：規定【ローカルレベルモデル、パラメータが既知】

data{
  int<lower=1>   t_max;    // 時系列長
  vector[t_max]   y;       // 観測値

  cov_matrix[1]   W;       // 状態雑音の分散
  cov_matrix[1]   V;       // 観測雑音の分散
  real           m0;       // 事前分布の平均
  cov_matrix[1]  C0;       // 事前分布の分散
}

parameters{
  real           x0;       // 状態[0]
  vector[t_max]   x;       // 状態[1:t_max]
}

model{
  // 尤度の部分
  /* 観測方程式・・・(5.11)式を参照 */
  for (t in 1:t_max){
    y[t] ~ normal(x[t], sqrt(V[1, 1]));
  }

  // 事前分布の部分
  /* 状態の事前分布 */
  x0   ~ normal(m0, sqrt(C0[1, 1]));

  /* 状態方程式・・・(5.10)式を参照 */
  x[1] ~ normal(x0, sqrt(W[1, 1]));
  for (t in 2:t_max){
    x[t] ~ normal(x[t-1], sqrt(W[1, 1]));
  }
}
