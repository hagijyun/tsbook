data {
	int<lower=0> N;
	int<lower=0> M;
	matrix[N, M] X;
	real<lower=0> y[N];
}

parameters {
	real s_q;
	vector<lower=0>[M] s_b;
	matrix<lower=0>[N, M] beta;
	real d_int;
}

model {
	for (i in 2:N){
		for (j in 1:M){
			beta[i, j] ~ normal(beta[i-1, j], s_b[j]);
		}
	}
	for (i in 1:N)
		y[i]~normal(dot_product(X[i,], beta[i,]) + d_int, s_q);
}
