// model10-3.stan
// モデル：規定【ローカルレベルモデル、パラメータが未知、カルマンフィルタを活用】

data{
  int<lower=1>    t_max;   // 時系列長
  matrix[1, t_max]    y;   // 観測値

  matrix[1, 1]    G;       // 状態遷移行列
  matrix[1, 1]    F;       // 観測行列
  vector[1]      m0;       // 事前分布の平均
  cov_matrix[1]  C0;       // 事前分布の分散
}

parameters{
  cov_matrix[1]   W;       // 状態雑音の分散
  cov_matrix[1]   V;       // 観測雑音の分散
}

model{
  // 尤度の部分
  /* 線形・ガウス型状態空間モデルの尤度を求める関数 */
  y ~ gaussian_dlm_obs(F, G, V, W, m0, C0);

  // 事前分布の部分
  /* W, Vの事前分布：無情報事前分布（省略時のデフォルト設定を活用） */
}
