import numpy as np

from model.model import Model
from model.init import uniform, constant
from model.activation import relu, softmax, relu_bp
from model.loss import categorical_crossentropy


class MLP(Model):
    def __init__(self, dim_in, hidden_units, dim_out, seed=None):
        super().__init__()
        assert len(hidden_units) > 0
        if seed is not None:
            np.random.seed(seed)
        dims = [dim_in] + hidden_units + [dim_out]
        self.ws, self.bs = [], []
        for i in range(len(dims) - 1):
            self.ws.append(uniform((dims[i], dims[i + 1])))
            self.bs.append(constant((dims[i + 1], )))

        self.zs, self.hs = [], []

    def _linear(self, h, w, b, activation=relu):
        self.hs.append(h)
        z = np.matmul(h, w) + b
        self.zs.append(z)
        h_n = activation(z)
        return h_n

    def forward(self, x):
        self.zs, self.hs = [], []
        output = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            output = self._linear(output, w, b)
        output = self._linear(output, self.ws[-1], self.bs[-1], softmax)
        return output

    def loss(self, y_hat, y):
        return categorical_crossentropy(y_hat, y, aggregation='mean')

    def backward(self, y_hat, y):
        n = len(y)
        dws, dbs = [], []
        phi_t = y_hat.copy()
        phi_t[np.arange(len(y)), y] -= 1
        for t in range(len(self.zs) - 1, -1, -1):
            h_t, w_t = self.hs[t], self.ws[t]
            dw = np.matmul(h_t.T, phi_t) / n
            db = np.mean(phi_t, axis=0)
            if t > 0:
                z_t = self.zs[t - 1]
                phi_t = np.matmul(phi_t, w_t.T) * relu_bp(z_t)
            dws.append(dw)
            dbs.append(db)
        return dws[::-1], dbs[::-1]


class MLP_Adam:
    def __init__(self, model, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.model = model

        assert learning_rate > 0
        assert 0 <= beta_1 < 1
        assert 0 <= beta_2 < 1
        assert eps > 0

        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.current_it = 0
        self.ms, self.vs = self._init_moment()

    def _init_moment(self):
        ms = [0] * (len(self.model.ws) + len(self.model.bs))
        vs = [0] * (len(self.model.ws) + len(self.model.bs))
        return ms, vs

    def _update_moment_m(self, ms, grads):
        n_ms = [self.beta_1 * m_t + (1 - self.beta_1) * grad
                   for m_t, grad in zip(ms, grads)]
        return n_ms

    def _update_moment_v(self, vs, grads):
        n_vs = [self.beta_2 * v_t + (1 - self.beta_2) * grad * grad
                   for v_t, grad in zip(vs, grads)]
        return n_vs

    def _bias_corrected(self, moment, beta):
        hat = [m / (1 - beta ** (self.current_it)) for m in moment]
        return hat

    def step(self, grads):
        self.current_it += 1
        dws, dbs = grads
        self.ms = self._update_moment_m(self.ms, dws + dbs)
        self.vs = self._update_moment_v(self.vs, dws + dbs)

        ms_hat = self._bias_corrected(self.ms, self.beta_1)
        vs_hat = self._bias_corrected(self.vs, self.beta_2)

        params = self.model.ws + self.model.bs
        n_params = [p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                       for p, m_hat, v_hat in zip(params, ms_hat, vs_hat)]
        split = len(self.model.ws)
        self.model.ws = n_params[:split]
        self.model.bs = n_params[split:]
