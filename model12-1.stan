// mode12-1.stan
// モデル：規定【ローカルレベルモデル＋馬蹄分布、パラメータが未知、時変カルマンフィルタを活用】

/*
 * 冒頭の functions ブロックの中身は下記サイトから（最新のrstanの警告を回避するため一部修正）
 * https://raw.githubusercontent.com/jrnold/dlm-shrinkage/master/stan/includes/dlm.stan
 */
 
functions{
  // -*- mode: stan -*- 
  // --- BEGIN: dlm ---
  matrix make_symmetric_matrix(matrix x) {
    return 0.5 * (x + x');
  }

  int dlm_filter_return_size(int r, int p) {
    int ret;
    ret = 2 * p * p + 2 * p + 2 * r;
    return ret;
  }

  // Convert vector to matrix of size
  // assumes that vector is in col-major order
  matrix to_matrix_wdim(vector v, int r, int c) {
    matrix[r, c] res;
    for (j in 1:c) {
      for (i in 1:r) {
        res[i, j] = v[(j - 1) * r + i];
      }
    }
    return res;
  }

  /*
  Kalman Filter

  - int n: number of observations
  - int r: number of variables
  - int p: number of states
  - vector[n] y: observations
  - int miss[r, n]: 1 if y is missing, 0 otherwise
  - vector[n] b: See definition
  - matrix[p] F[n]
  - vector[p] V[n]
  - vector[p] g[n]
  - vector[p, p] G[n]
  - vector[p, p] W[n]
  - vector[p] m0
  - matrix[p, p] C0


  Returns

  `vector[2 p^2 + 2 p + 2] ret[n + 1]`: Each element in the array is a time period.
   The first element is $t = 0$ for the initial values of $m$ and $C$, the 2nd element
   is $t = 1$ and so on. Each vector is $[m_t, C_t, a_t, R_t, e, Q^{-1}]$,
   where matrices are converted to vecctors as column-major using `to_vector`.
   The indices of these elements are:

   - $[1, p]$: $m_t$
   - $[p + 1, p + p ^ 2]$: $C_t$
   - $[p + p^2 + 1, 2 p + p^2]$: $a_t$
   - $[2p + p^2 + 1, 2 p + 2p^2]$: $R_t$
   - $2p + 2p^2 + 1$: $f_t$
   - $2p + 2p^2 + 2$: $Q^{-1}_t$

  Each vector is $2 p^2 + 2 p + 2$ in

  */
  vector[] dlm_filter(int n, int r, int p,
                        vector[] y, int[,] miss,
                        vector[] b, matrix[] F, vector[] V,
                        vector[] g, matrix[] G, matrix[] W,
                        vector m0, matrix C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    vector[p] a;
    matrix[p, p] R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    vector[p] Fj;
    real f;
    real Q;
    vector[r] Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    vector[p] m;
    matrix[p, p] C;
    // forecast error
    vector[r] e;
    // Kalman gain
    vector[p] K;
    // returned data
    vector[dlm_filter_return_size(r, p)] res[n + 1];
    vector[p * p] C_vec;
    vector[p * p] R_vec;

    // print("g = ", g);
    //   print("G = ", G);
    //   print("b = ", b);
    //   print("F = ", F);
    //   print("V = ", V);
    //   print("W = ", W);
    //   print("m0 = ", m0);
    //   print("C0 = ", C0);
    //   print("y = ", y);
    //   print("miss = ", miss);

    
    m = m0;
    C = C0;
    // print("t=", 0, ", m=", m);
    // print("t=", 0, ", C=", C);

    res[1] = rep_vector(0.0, dlm_filter_return_size(r, p));
    for(i in 1:p) {
      res[1, i] = m[i];
    }
    C_vec = to_vector(C);
    for (i in 1:(p * p)) {
      res[1, p + i] = C_vec[i];
    }
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = g[t] + G[t] * m;
      R = quad_form(C, G[t] ') + W[t];
      m = a;
      C = R;
      // print("t=", t, ", a=", a);
      // print("t=", t, ", R=", R);
      for (j in 1:r) {
        if (int_step(miss[t, j])) {
          e[j] = 0.0;
          Q_inv[j] = 0.0;
        } else {
          // print("t = ", t, ", predict");
          // PREDICT OBS
          // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
          Fj = row(F[t], j) ';
          f = b[t, j] + dot_product(Fj, m);
          Q = quad_form(C, Fj) + V[t, j];
          Q_inv[j] = 1.0 / Q;
          // forecast error
          e[j] = y[t, j] - f;
          // Kalman gain
          K = C * Fj * Q_inv[j];
          // print("t = ", t, ", filter");
          // FILTER STATES
          // posterior distribution of p(\theta_t | y_{1:t})
          m = m + K * e[j];
          C = make_symmetric_matrix(C - K * Q * K ');
        }
        // print("t=", t, ", j=", j, ", m=", m);
        // print("t=", t, ", j=", j, ", C=", C);
        // print("t=", t, ", j=", j, ", Q_inv=", Q_inv);
        // print("t=", t, ", j=", j, ", f=", f);
        // print(" ");
      }
      for(i in 1:p) {
        res[t + 1, i] = m[i];
      }
      C_vec = to_vector(C);
      for (i in 1:(p * p)) {
        res[t + 1, p + i] = C_vec[i];
      }
      for(i in 1:p) {
        res[t + 1, p + p * p + i] = a[i];
      }
      R_vec = to_vector(R);
      for (i in 1:(p * p)) {
        res[t + 1, 2 * p + p * p + i] = R_vec[i];
      }
      for (i in 1:r) {
        res[t + 1, 2 * p + 2 * p * p + i] = e[i];
      }
      for (i in 1:r) {
        res[t + 1, 2 * p + 2 * p * p + r + i] = Q_inv[i];
      }
    }
    return res;
  }


  vector[] dlm_constant_filter(int n, int r, int p,
                               vector[] y, int[,] miss,
                               vector b, matrix F, vector V,
                               vector g, matrix G, matrix W,
                               vector m0, matrix C0) {
    vector[r] b_tv[n];
    matrix[r, p] F_tv[n];
    vector[r] V_tv[n];
    vector[p] g_tv[n];
    matrix[p, p] G_tv[n];
    matrix[p, p] W_tv[n];

    b_tv = rep_array(b, n);
    F_tv = rep_array(F, n);
    V_tv = rep_array(V, n);  
    g_tv = rep_array(g, n);
    G_tv = rep_array(G, n);
    W_tv = rep_array(W, n);  
    return dlm_filter(n, r, p, y, miss,
                      b_tv, F_tv, V_tv,
                      g_tv, G_tv, W_tv,
                      m0, C0);
  }


  vector[] dlm_loglik(int n, int r, int p,
                        vector[] y, int[,] miss,
                        vector[] b, matrix[] F, vector[] V,
                        vector[] g, matrix[] G, matrix[] W,
                        vector m0, matrix C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    vector[p] a;
    matrix[p, p] R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    vector[p] Fj;
    real f;
    real Q;
    real Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    vector[p] m;
    matrix[p, p] C;
    // forecast error
    real e;
    // Kalman gain
    vector[p] K;
    // Loglik
    vector[r] loglik[n];

    m = m0;
    C = C0;
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = g[t] + G[t] * m;
      R = quad_form(C, G[t] ') + W[t];
      m = a;
      C = R;
      for (j in 1:r) {
        if (int_step(miss[t, j])) {
          loglik[t, j] = 0.0;
        } else {
          // PREDICT OBS
          // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
          Fj = F[t, j] ';
          f = b[t, j] + dot_product(Fj, m);
          Q = quad_form(C, Fj) + V[t, j];
          Q_inv = 1.0 / Q;
          // forecast error
          e = y[t, j] - f;
          // Kalman gain
          K = C * Fj * Q_inv;
          // FILTER STATES
          // posterior distribution of p(\theta_t | y_{1:t})
          m = m + K * e;
          C = make_symmetric_matrix(C - K * Q * K ');
          loglik[t, j] = - 0.5 * (log(2 * pi()) + log(Q) + e ^ 2 * Q_inv);
        }
      }
    }
    return loglik;
  }

  vector[] dlm_constant_loglik(int n, int r, int p,
                               vector[] y, int[,] miss,
                               vector b, matrix F, vector V,
                               vector g, matrix G, matrix W,
                               vector m0, matrix C0) {
    vector[r] b_tv[n];
    matrix[r, p] F_tv[n];
    vector[r] V_tv[n];
    vector[p] g_tv[n];
    matrix[p, p] G_tv[n];
    matrix[p, p] W_tv[n];

    b_tv = rep_array(b, n);
    F_tv = rep_array(F, n);
    V_tv = rep_array(V, n);  
    g_tv = rep_array(g, n);
    G_tv = rep_array(G, n);
    W_tv = rep_array(W, n);  
    return dlm_loglik(n, r, p, y, miss,
                      b_tv, F_tv, V_tv,
                      g_tv, G_tv, W_tv,
                      m0, C0);
  }


  // directly increment log-probability without returning the individual log-likelihoods
  real dlm_logpdf(int n, int r, int p,
                  vector[] y, int[,] miss,
                  vector[] b, matrix[] F, vector[] V,
                  vector[] g, matrix[] G, matrix[] W,
                  vector m0, matrix C0) {
    vector[r] llall[n];
    vector[n] llobs;
    real ll;
    llall = dlm_loglik(n, r, p, y, miss,
                     b, F, V,
                     g, G, W,
                     m0, C0);
    for (i in 1:n) {
      llobs[i] = sum(llall[i]);
    }
    ll = sum(llobs);
    return ll;
  }

  real dlm_constant_log(int n, int r, int p,
                        vector[] y, int[,] miss,
                        vector b, matrix F, vector V,
                        vector g, matrix G, matrix W,
                        vector m0, matrix C0) {
    vector[r] b_tv[n];
    matrix[r, p] F_tv[n];
    vector[r] V_tv[n];
    vector[p] g_tv[n];
    matrix[p, p] G_tv[n];
    matrix[p, p] W_tv[n];

    b_tv = rep_array(b, n);
    F_tv = rep_array(F, n);
    V_tv = rep_array(V, n);  
    g_tv = rep_array(g, n);
    G_tv = rep_array(G, n);
    W_tv = rep_array(W, n);  
    return dlm_logpdf(n, r, p, y, miss,
                   b_tv, F_tv, V_tv,
                   g_tv, G_tv, W_tv,
                   m0, C0);
  }


  void dlm_lp(int n, int r, int p, 
                  vector[] y, int[,] miss,
                  vector[] b, matrix[] F, vector[] V,
                  vector[] g, matrix[] G, matrix[] W,
                  vector m0, matrix C0) {
    target += dlm_logpdf(n, r, p, y, miss, b, F, V, g, G, W, m0, C0);
  }


  void dlm_constant_lp(int n, int r, int p,
                        vector[] y, int[,] miss,
                        vector b, matrix F, vector V,
                        vector g, matrix G, matrix W,
                        vector m0, matrix C0) {
    vector[r] b_tv[n];
    matrix[r, p] F_tv[n];
    vector[r] V_tv[n];
    vector[p] g_tv[n];
    matrix[p, p] G_tv[n];
    matrix[p, p] W_tv[n];

    b_tv = rep_array(b, n);
    F_tv = rep_array(F, n);
    V_tv = rep_array(V, n);  
    g_tv = rep_array(g, n);
    G_tv = rep_array(G, n);
    W_tv = rep_array(W, n);  
    dlm_lp(n, r, p, y, miss,
           b_tv, F_tv, V_tv,
           g_tv, G_tv, W_tv,
           m0, C0);
  }


  vector dlm_get_m(int t, int r, int p, vector[] v) {
    vector[p] x;
    x = segment(v[t + 1], 1, p);
    return x;
  }


  vector[] dlm_get_m_all(int n, int r, int p, vector[] v) {
    vector[p] res[n + 1];
    for (t in 0:n) {
      res[t + 1] = dlm_get_m(t, r, p, v);
    }
    return res;
  }


  matrix dlm_get_C(int t, int r, int p, vector[] v) {
    vector[p * p] x;
    matrix[p, p] res;
    x = segment(v[t + 1], p + 1, p * p);
    res = to_matrix_wdim(x, p, p);
    return res;
  }


  matrix[] dlm_get_C_all(int n, int r, int p, vector[] v) {
    matrix[p, p] res[n + 1];
    for (t in 0:n) {
      res[t + 1] = dlm_get_C(t, r, p, v);
    }
    return res;
  }


  vector dlm_get_a(int t, int r, int p, vector[] v) {
    vector[p] res;
    res = segment(v[t + 1], p + p * p + 1, p);
    return res;
  }


  vector[] dlm_get_a_all(int n, int r, int p, vector[] v) {
    vector[p] res[n];
    for (t in 1:n) {
      res[t] = dlm_get_a(t, r, p, v);
    }
    return res;
  }


  matrix dlm_get_R(int t, int r, int p, vector[] v) {
    vector[p * p] x;
    matrix[p, p] res;
    x = segment(v[t + 1], 2 * p + p * p + 1, p * p);
    res = to_matrix_wdim(x, p, p);
    return res;
  }


  matrix[] dlm_get_R_all(int n, int r, int p, vector[] v) {
    matrix[p, p] res[n];
    for (t in 1:n) {
      res[t] = dlm_get_R(t, r, p, v);
    }
    return res;
  }


  vector dlm_get_e(int t, int r, int p, vector[] v) {
    vector[r] res;
    res = segment(v[t + 1], 2 * p + 2 * p * p + 1, r);
    return res;
  }


  vector[] dlm_get_e_all(int n, int r, int p, vector[] v) {
    vector[r] res[n];
    for (t in 1:n) {
      res[t] = dlm_get_e(t, r, p, v);
    }
    return res;
  }


  vector dlm_get_Q_inv(int t, int r, int p, vector[] v) {
    vector[r] res;
    res = segment(v[t + 1], 2 * p + 2 * p * p + r + 1, r);
    return res;
  }


  vector[] dlm_get_Q_inv_all(int n, int r, int p, vector[] v) {
    vector[r] res[n];
    for (t in 1:n) {
      res[t] = dlm_get_Q_inv(t, r, p, v);
    }
    return res;
  }

  vector[] dlm_filter_loglik(int n, int r, int p, vector[] v, int[,] miss) {
    vector[r] ll[n];
    vector[r] err;
    vector[r] Q_inv;
    for (i in 1:n) {
      err = dlm_get_e(i, r, p, v);
      Q_inv = dlm_get_Q_inv(i, r, p, v);
      for (j in 1:r) {
        if (int_step(miss[i, j])) {
          ll[i, j] = 0.0;
        } else {
          ll[i, j] =  - 0.5 * (log(2 * pi()) - log(Q_inv[j]) + (err[j] ^ 2) * Q_inv[j]);
        }
      }
    }
    return ll;
  }

  /*
  Backward sampling given results of a Kalman Filter

  - int n: Number of observations
  - int p: Number of states
  - matrix[p, p] G[n]: Transition matrices
  - vector[p] m[n + 1]: Mean of filtered states $p(\theta_t | y_1, \dots, y_t)$.
    Note that there are $n + 1$ entries, because $i = 1$ corresponds to the initial
    values at $t = 0$.
  - matrix[p, p] C[n + 1]: Covariance of filtered states $Cov(\theta_t | y_1, \dots, y_{t - 1})$.
  - vector[p] a[n]: Mean of predicted states $p(\theta_t \ y_1, \dots, y_t)$.
  - matrix[p, p] R[n]: Covariance of predicted states $Cov(\theta_t | y_t, \dots, y_{t - 1})$.

  */
  vector[] dlm_bsample_rng(int n, int p,
                                    matrix[] G,
                                    vector[] m, matrix[] C,
                                    vector[] a, matrix[] R) {
    vector[p] theta[n + 1];
    vector[p] h;
    matrix[p, p] H;
    matrix[p, p] R_inv;
    // TODO: sample once and cholesky transform

    // set last state
    theta[n + 1] = multi_normal_rng(m[n + 1], C[n + 1]);
    for (i in 0:(n - 1)) {
       int t;
       t = n - i;
       R_inv = inverse(R[t]);
       h = m[t] + C[t] * G[t] ' * R_inv  * (theta[t + 1] - a[t]);
       H = make_symmetric_matrix(C[t] - C[t] * quad_form(R_inv, G[t]) * C[t]);
       theta[t] = multi_normal_rng(h, H);
    }
    return theta;
  }

  vector[] dlm_filter_bsample_rng(int n, int r, int p, 
                                  matrix[] G,
                                  vector[] v) {
    vector[p] theta[n + 1];
    vector[p] h;
    matrix[p, p] H;
    matrix[p, p] R_inv;
    vector[p] a;
    matrix[p, p] R;
    vector[p] m;
    matrix[p, p] C;
    matrix[p, p] G_t;

    // TODO: sample once and cholesky transform
    // set last state
    m = dlm_get_m(n, r, p, v);
    C = dlm_get_C(n, r, p, v);
    theta[n + 1] = multi_normal_rng(m, C);
    for (i in 1:n) {
       int t;
       t = n - i;
       m = dlm_get_m(t, r, p, v);
       C = dlm_get_C(t, r, p, v);
       a = dlm_get_a(t + 1, r, p, v);
       R = dlm_get_R(t + 1, r, p, v);
       G_t = G[t + 1];

       R_inv = inverse(R);
       h = m + C * G_t ' * R_inv  * (theta[t + 2] - a);
       H = make_symmetric_matrix(C - C * quad_form(R_inv, G_t) * C);
       theta[t + 1] = multi_normal_rng(h, H);
    }
    return theta;
  }


  // smoother
  vector[] dlm_smooth(int n, int p,
                      matrix[] G,
                      vector[] m, matrix[] C,
                      vector[] a, matrix[] R) {
    vector[p] theta[n + 1];
    vector[p] s;
    matrix[p, p] S;
    vector[p * p] S_vec;
    matrix[p, p] R_inv;
    // TODO: check dim;
    vector[p * p + p] ret[n + 1];

    // set last state
    s = m[n + 1];
    S = C[n + 1]; 
    for (i in 1:p) {
      ret[n + 1, i] = s[i];
    }
    S_vec = to_vector(S);
    for (i in 1:(p * p)) {
      ret[n + 1, p + i] = S_vec[i];
    }
    for (i in 0:(n - 1)) {
       int t;
       t = n - i;
       R_inv = inverse(R[t]);
       s = m[t] + C[t] * G[t] ' * R_inv  * (s - a[t]);
       S = make_symmetric_matrix(C[t] - C[t] * G[t] ' * (R[t] - S) * R_inv * G[t] * C[t]);
       for (j in 1:p) {
         ret[t, j] = s[i];
       }
       S_vec = to_vector(S);
       for (j in 1:(p * p)) {
         ret[t, p + j] = S_vec[i];
       }
    }
    return ret;
  }

  vector dlm_smooth_get_s(int t, int p, vector[] v) {
    vector[p] ret;
    ret = segment(v[t + 1], 1, p);
    return ret;
  }

  vector[] dlm_smooth_get_s_all(int n, int p, vector[] v) {
    vector[p] ret[n + 1];
    for (t in 0:n) {
       ret[t + 1] = dlm_smooth_get_s(t, p, v);
    }
    return ret;
  }

  matrix dlm_smooth_get_S(int t, int p, vector[] v) {
    vector[p * p] x;
    matrix[p, p] res;
    x = segment(v[t + 1], p + 1, p * p);
    res = to_matrix_wdim(x, p, p);
    return res;
  }

  matrix[] dlm_smooth_get_S_all(int n, int p, vector[] v) {
    matrix[p, p] res[n + 1];
    for (t in 0:n) {
      res[t + 1] = dlm_smooth_get_S(t, p, v);
    }
    return res;
  }



  // DLM calculated with discounting
  vector[] dlm_discount_filter(int n, int r, int p,
                               vector[] y, int[,] miss,
                               vector[] b, matrix[] F, vector[] V,
                               vector[] g, matrix[] G, vector[] d,
                               vector m0, matrix C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    vector[p] a;
    matrix[p, p] R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    vector[p] Fj;
    real f;
    real Q;
    vector[r] Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    vector[p] m;
    matrix[p, p] C;
    // forecast error
    vector[r] e;
    // Kalman gain
    vector[p] K;
    // returned data
    vector[dlm_filter_return_size(r, p)] res[n + 1];
    vector[p * p] C_vec;
    vector[p * p] R_vec;

    m = m0;
    C = C0;
    res[1] = rep_vector(0.0, dlm_filter_return_size(r, p));
    for(i in 1:p) {
      res[1, i] = m[i];
    }
    C_vec = to_vector(C);
    for (i in 1:(p * p)) {
      res[1, p + i] = C_vec[i];
    }
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = g[t] + G[t] * m;
      R = diag_pre_multiply(1.0 ./ d[t], quad_form(C, G[t] '));
      m = a;
      C = R;
      for (j in 1:r) {
        if (int_step(miss[t, j])) {
          e[j] = 0.0;
          Q_inv[j] = 0.0;
        } else {
          // PREDICT OBS
          // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
          Fj = F[t, j] ';
          f = b[t, j] + dot_product(Fj, m);
          Q = quad_form(C, Fj) + V[t, j];
          Q_inv[j] = 1.0 / Q;
          // forecast error
          e[j] = y[t, j] - f;
          // Kalman gain
          K = C * Fj * Q_inv[j];
          // FILTER STATES
          // posterior distribution of p(\theta_t | y_{1:t})
          m = m + K * e[j];
          C = make_symmetric_matrix(C - K * Q * K ');
        }
      }
      for(i in 1:p) {
        res[t + 1, i] = m[i];
      }
      C_vec = to_vector(C);
      for (i in 1:(p * p)) {
        res[t + 1, p + i] = C_vec[i];
      }
      for(i in 1:p) {
        res[t + 1, p + p * p + i] = a[i];
      }
      R_vec = to_vector(R);
      for (i in 1:(p * p)) {
        res[t + 1, 2 * p + p * p + i] = R_vec[i];
      }
      for (i in 1:r) {
        res[t + 1, 2 * p + 2 * p * p + i] = e[i];
      }
      for (i in 1:r) {
        res[t + 1, 2 * p + 2 * p * p + r + i] = Q_inv[i];
      }
    }
    return res;
  }


  vector[] dlm_discount_loglik(int n, int r, int p, 
                               vector[] y, int[,] miss,
                               vector[] b, matrix[] F, vector[] V,
                               vector[] g, matrix[] G, vector[] d,
                               vector m0, matrix C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    vector[p] a;
    matrix[p, p] R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    vector[p] Fj;
    real f;
    real Q;
    real Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    vector[p] m;
    matrix[p, p] C;
    // forecast error
    real e;
    // Kalman gain
    vector[p] K;
    // Loglik
    vector[r] loglik[n];

    m = m0;
    C = C0;
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = g[t] + G[t] * m;
      R = diag_pre_multiply(1.0 ./ d[t], quad_form(C, G[t] '));
      m = a;
      C = R;
      for (j in 1:r) {
        if (int_step(miss[t, j])) {
          loglik[t, j] = 0.0;
        } else {
          // PREDICT OBS
          // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
          Fj = F[t, j] ';
          f = b[t, j] + dot_product(Fj, m);
          Q = quad_form(C, Fj) + V[t, j];
          Q_inv = 1.0 / Q;
          // forecast error
          e = y[t, j] - f;
          // Kalman gain
          K = C * Fj * Q_inv;
          // FILTER STATES
          // posterior distribution of p(\theta_t | y_{1:t})
          m = m + K * e;
          C = make_symmetric_matrix(C - K * Q * K ');
          loglik[t, j] = - 0.5 * (log(2 * pi()) + log(Q) + e ^ 2 * Q_inv);
        }
      }
    }
    return loglik;
  }

  real dlm_discount_logpdf(int n, int r, int p, 
                  vector[] y, int[,] miss,
                  vector[] b, matrix[] F, vector[] V,
                  vector[] g, matrix[] G, vector[] d,
                  vector m0, matrix C0) {
    vector[r] llall[n];
    vector[n] llobs;
    real ll;
    llall = dlm_discount_loglik(n, r, p, y, miss,
                                 b, F, V,
                                 g, G, d,
                                 m0, C0);
    for (i in 1:n) {
      llobs[i] = sum(llall[i]);
    }
    ll = sum(llobs);
    return ll;
  }


  void dlm_discount_lp(int n, int r, int p, 
                  vector[] y, int[,] miss,
                  vector[] b, matrix[] F, vector[] V,
                  vector[] g, matrix[] G, vector[] d,
                  vector m0, matrix C0) {
    target += dlm_discount_logpdf(n, r, p, y, miss, b, F, V, g, G, d, m0, C0);
  }



  // Univariate Random Walk -------------------------------

  vector[] dlm_local_level_filter(int n,
                                  vector y, int[] miss,
                                  vector V, vector W,
                                  real m0, real C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    real a;
    real R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    real f;
    real Q;
    real Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    real m;
    real C;
    // forecast error
    real e;
    // Kalman gain
    real K;
    // returned data
    vector[dlm_filter_return_size(1, 1)] res[n + 1];

    m = m0;
    C = C0;
    res[1] = rep_vector(0.0, dlm_filter_return_size(1, 1));
    res[1, 1] = m;
    res[1, 2] = C;
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = m;
      R = C + W[t];
      m = a;
      C = R;
      if (int_step(miss[t])) {
        e = 0.0;
        Q_inv = 0.0;
      } else {
        // PREDICT OBS
        // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
        f = m;
        Q = C + V[t];
        Q_inv = 1.0 / Q;
        // forecast error
        e = y[t] - f;
        // Kalman gain
        K = C * Q_inv;
        // FILTER STATES
        // posterior distribution of p(\theta_t | y_{1:t})
        m = m + K * e;
        C = C - pow(K, 2) * Q;
      }
      res[t + 1, 1] = m;
      res[t + 1, 2] = C;
      res[t + 1, 3] = a;
      res[t + 1, 4] = R;
      res[t + 1, 5] = e;
      res[t + 1, 6] = Q_inv;
    }
    return res;
  }

  vector dlm_local_level_loglik(int n,
                                vector y, int[] miss,
                                vector V, vector W,
                                real m0, real C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    real a;
    real R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    real f;
    real Q;
    real Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    real m;
    real C;
    // forecast error
    real e;
    // Kalman gain
    real K;
    // returned data
    vector[n] loglik;

    m = m0;
    C = C0;
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = m;
      R = C + W[t];
      m = a;
      C = R;
      if (int_step(miss[t])) {
        e = 0.0;
        Q_inv = 0.0;
      } else {
        // PREDICT OBS
        // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
        f = m;
        Q = C + V[t];
        Q_inv = 1.0 / Q;
        // forecast error
        e = y[t] - f;
        // Kalman gain
        K = C * Q_inv;
        // FILTER STATES
        // posterior distribution of p(\theta_t | y_{1:t})
        m = m + K * e;
        C = C - pow(K, 2) *  Q;
        loglik[t] = - 0.5 * (log(2 * pi()) + log(Q) + pow(e, 2) * Q_inv);
      }
    }
    return loglik;
  }

  vector dlm_local_level_filter_loglik(int n, vector[] v, int[] miss) {
    vector[n] ll;
    vector[1] err;
    vector[1] Q_inv;
    for (i in 1:n) {
      err = dlm_get_e(i, 1, 1, v);
      Q_inv = dlm_get_Q_inv(i, 1, 1, v);
      if (int_step(miss[i])) {
        ll[i] = 0.0;
      } else {
        ll[i] =  - 0.5 * (log(2 * pi()) - log(Q_inv[1]) + (err[1] ^ 2) * Q_inv[1]);
      }
    }
    return ll;
  }



  real dlm_local_level_log(int n,
                           vector y, int[] miss,
                           vector V, vector W,
                           real m0, real C0) {
    real ll;
    ll = sum(dlm_local_level_loglik(n, y, miss,
                                     V, W, m0, C0));
    return ll;
  }

  void dlm_local_level_lp(int n,
                            vector y, int[] miss,
                            vector V, vector W,
                            real m0, real C0) {
    target += dlm_local_level_loglik(n, y, miss,
                                              V, W, m0, C0);
                                              
  }

  // --- END: dlm ---
}

data{
  int<lower = 1>              t_max;  // 時系列長
  vector[t_max]                   y;  // 観測値
  
  int                   miss[t_max];  // 欠測値の指標
  real                           m0;  // 事前分布の平均
  real<lower = 0>                C0;  // 事前分布の分散
}

parameters{
  vector<lower = 0>[t_max]   lambda;  // 状態雑音の標準偏差（時　変の倍率　）
    real<lower = 0>          W_sqrt;  // 状態雑音の標準偏差（時不変のベース）
    real<lower = 0>          V_sqrt;  // 観測雑音の標準偏差（時不変　　　　）
}

transformed parameters{
  vector[t_max]                   W;  // 状態雑音の分散（時変）
  vector[t_max]                   V;  // 観測雑音の分散（時変）
  real                      log_lik;  // 全時点分の対数尤度

  // 状態雑音の分散（全時点で異なる値が設定される）
  for (t in 1:t_max) {
    W[t] = pow(lambda[t] * W_sqrt, 2);
  }

  // 観測雑音の分散（全時点で　同じ値が設定される）
  V = rep_vector(pow(V_sqrt, 2), t_max);

  // 最終的な対数尤度
  log_lik = sum(dlm_local_level_loglik(t_max, y, miss, V, W, m0, C0));
}

model{
  // 尤度の部分
  target += log_lik;

  // 事前分布の部分
  /* 状態雑音の標準偏差（時変の倍率） */
  lambda ~ cauchy(0, 1);

  /* 状態雑音の標準偏差（時不変のベース）、観測雑音の標準偏差（時不変）：無情報事前分布 */
}
