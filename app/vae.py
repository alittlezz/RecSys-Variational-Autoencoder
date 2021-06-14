import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable as V


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def weights_init(m):
    pass
    # if isinstance(m, nn.Linear):
    #     torch.nn.init.normal_(m.weight, 0, 0.01)
    #     torch.nn.init.normal(m.bias, 0, 0.01)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(input_dim, 40)
        self.ln1 = nn.LayerNorm(40, eps=eps)

        self.fc_mu = nn.Linear(40, latent_dim)
        self.fc_logvar = nn.Linear(40, latent_dim)

        self.fc1.apply(weights_init)
        self.fc_mu.apply(weights_init)
        self.fc_logvar.apply(weights_init)

    def forward(self, x, dropout_rate, calculate_loss=True):
        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(self.act(self.fc1(x)))

        mu, logvar = self.fc_mu(h1), self.fc_logvar(h1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Decoder, self).__init__()

        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(latent_dim, input_dim)

        self.fc1.apply(weights_init)

    def forward(self, z, calculate_loss=True):
        x = torch.sigmoid(self.fc1(z))

        return x


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, device):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, input_dim)
        self.zr = torch.zeros(1, latent_dim).to(device=device)
        self.mones = torch.ones(1, latent_dim).to(device) * -1

    def forward(self, user_ratings, alpha=0.5, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True, n_epoch=1):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate, calculate_loss=calculate_loss)
        z = reparameterize(mu, logvar)
        x_pred = self.decoder(z, calculate_loss=calculate_loss)

        if calculate_loss:
            if gamma:
                kl_weight = gamma * 10
            elif beta:
                kl_weight = beta

            # Choose between MSE and cross-entropy

            # user_mask = user_ratings > 0
            # mll = torch.pow(x_pred * user_mask - user_ratings, 2).sum(dim=-1).mul(kl_weight).mean()
            mll = (-torch.log(x_pred) * user_ratings).sum(dim=-1).mul(kl_weight).mean()

            # Compute divergence
            prior = log_norm_pdf(z, self.zr, self.zr)
            kld = (log_norm_pdf(z, mu, logvar) - prior).sum(dim=-1).mul(kl_weight).mean()

            negative_elbo = mll + kld

            return mu, z, negative_elbo
        else:
            return mu

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


class DualVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim_users, input_dim_movies, device, eps=1e-1):
        super(DualVAE, self).__init__()
        self.device = device
        self.movies_VAE = VAE(hidden_dim, latent_dim, input_dim_users, device)
        self.users_VAE = VAE(hidden_dim, latent_dim, input_dim_movies, device)
        self.input_dim_users = input_dim_users
        self.input_dim_movies = input_dim_movies
        self.f = latent_dim
        self.f_eye = torch.eye(self.f).to(device)
        self.lambda_u = 10
        self.lambda_v = 10
        mU = MultivariateNormal(torch.zeros(latent_dim), 1 / self.lambda_u * torch.eye(latent_dim))
        mV = MultivariateNormal(torch.zeros(latent_dim), 1 / self.lambda_v * torch.eye(latent_dim))
        self.U = torch.stack([mU.sample() for _ in range(input_dim_users)]).to(device)
        self.V = torch.stack([mV.sample() for _ in range(input_dim_movies)]).to(device)

    def pmf_init(self, R_train, R_train_tensor, R_train_T, R_train_tensor_T):
        # Compute initial U vector
        self.v_conv_all = torch.stack([torch.einsum('n,m->nm', vj, vj.T) for vj in self.V])

        self.u_dev = torch.zeros((self.input_dim_users, self.f, self.f)).to(self.device)
        for i in range(self.input_dim_users):
            batch = torch.from_numpy(R_train[i].nonzero()[1]).to(dtype=torch.int64)
            v_sum = torch.sum(self.v_conv_all[batch], axis=0)
            self.u_dev[i] = torch.inverse(v_sum + self.lambda_u * self.f_eye)

            v_conv = torch.sum(R_train_tensor[i] * self.V[batch], axis=0)
            self.U[i] = self.u_dev[i] @ v_conv

        # Compute initial V vector
        self.u_conv_all = torch.stack([torch.einsum('n,m->nm', ui, ui.T) for ui in self.U])

        self.v_dev = torch.zeros((self.input_dim_movies, self.f, self.f)).to(self.device)
        for j in range(self.input_dim_movies):
            batch = torch.from_numpy(R_train_T[j].nonzero()[1]).to(dtype=torch.int64)
            u_sum = torch.sum(self.u_conv_all[batch], axis=0)
            self.v_dev[j] = torch.inverse(u_sum + self.lambda_v * self.f_eye)

            u_conv = torch.sum(R_train_tensor_T[j] * self.U[batch], axis=0)
            self.V[j] = self.v_dev[j] @ u_conv

    def forward(self, ratings, R_train, R_train_tensor, R_train_T, R_train_tensor_T, userFeatures=None, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True, n_epoch=1):
        if calculate_loss:
            mu_users, z_i, user_loss = self.users_VAE(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate,
                                                      calculate_loss=calculate_loss, n_epoch=n_epoch)
            mu_movies, z_j, movie_loss = self.movies_VAE(ratings.T, beta=beta, gamma=gamma, dropout_rate=dropout_rate,
                                                         calculate_loss=calculate_loss, n_epoch=n_epoch)
            self.pmf_estimate(mu_users, mu_movies, R_train, R_train_tensor, R_train_T, R_train_tensor_T)
            # y_hat = self.U @ self.V.T

            normalizer = 0.5 * self.lambda_u * torch.sum(
                self.u_dev * torch.pow(self.U - mu_users, 2).unsqueeze(2)) + 0.5 * self.lambda_v * torch.sum(
                self.v_dev * torch.pow(self.V - mu_movies, 2).unsqueeze(2))
            total_loss = user_loss + movie_loss + normalizer
            return total_loss, user_loss, movie_loss, normalizer

        if userFeatures is not None:
            mu_users = self.users_VAE(userFeatures, beta=beta, gamma=gamma, dropout_rate=dropout_rate,
                                      calculate_loss=calculate_loss, n_epoch=n_epoch)
            y_hat = mu_users @ self.V.T
        else:
            y_hat = self.U @ self.V.T

        return y_hat

    def pmf_estimate(self, mu_users, mu_movies, R_train, R_train_tensor, R_train_T, R_train_tensor_T):
        with torch.no_grad():
            # Update U vector
            self.v_conv_all = torch.stack([torch.einsum('n,m->nm', vj, vj.T) for vj in self.V])
            for i in range(self.input_dim_users):
                batch = torch.from_numpy(R_train[i].nonzero()[1]).to(dtype=torch.int64)
                v_sum = torch.sum(self.v_conv_all[batch], axis=0)
                v_dev_sm = torch.sum(self.v_dev[batch], axis=0)
                self.u_dev.data[i] = torch.inverse(v_sum + v_dev_sm + self.lambda_u * self.f_eye)

                v_conv = torch.sum(R_train_tensor[i] * self.V[batch], axis=0)
                self.U.data[i] = self.u_dev[i] @ (v_conv + self.lambda_u * mu_users[i])

            # Update V vector
            self.u_conv_all = torch.stack([torch.einsum('n,m->nm', ui, ui.T) for ui in self.U])
            for j in range(self.input_dim_movies):
                batch = torch.from_numpy(R_train_T[j].nonzero()[1]).to(dtype=torch.int64)
                u_sum = torch.sum(self.u_conv_all[batch], axis=0)
                u_dev_sm = torch.sum(self.u_dev[batch], axis=0)
                self.v_dev.data[j] = torch.inverse(u_sum + u_dev_sm + self.lambda_v * self.f_eye)

                u_conv = torch.sum(R_train_tensor_T[j] * self.U[batch], axis=0)
                self.V.data[j] = self.v_dev[j] @ (u_conv + self.lambda_v * mu_movies[j])

    def update_prior(self):
        self.users_VAE.update_prior()
        self.movies_VAE.update_prior()