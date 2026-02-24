# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: OpenSynth-BNsxhSIM
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Privacy evaluation
#
# This tutorial explains how to evaluate the privacy of synthetic smart meter data. This notebook is based on the Faraday architecture, trained on Low Carbon London dataset.
#
# - For more information on Faraday's architecture, refer to the [Faraday paper](https://arxiv.org/abs/2404.04314).
# - For more information on evaluation method, refer to the paper by Centre for Net Zero on [Defining 'Good': Evaluation Framework for Synthetic Smart Meter Data
# ](https://arxiv.org/abs/2407.11785).
#
# ### Pre-requisites
#
# 1. If you haven't already, please download LCL dataset from [data.london.gov.uk](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households).
# 2. If you haven't already, you need to first train a model (refer to the `faraday_tutorial.ipynb` notebook).
#

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import os
import sys

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# # 💿 Loading LCL Data

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.data_modules.pulse_data_module import PulseDataModule

data_path = Path("../../data/processed/historical/train/data.csv")
stats_path = Path("../../data/processed/historical/train/mean_std.csv")
outlier_path = Path("../../data/processed/historical/train/outliers.csv")

# Original training data with no outliers
dm = LCLDataModule(
    data_path=data_path, stats_path=stats_path, batch_size=200, n_samples=20000
)
dm.setup()

# Original training with PULSE data and NO outliers
# %%
prepped_path = "/home/llan/projects/alliander/GUIDE-VAE/preprocessed"
dm_pulse = PulseDataModule(prepped_path, batch_size=32)
# %%
# Training data with implausible outliers injected for privacy attacks
dm_with_outliers = LCLDataModule(
    data_path=data_path,
    stats_path=stats_path,
    batch_size=200,
    n_samples=20000,
    outlier_path=outlier_path,
)
dm_with_outliers.setup()

# %%
# Holdout data
holdout_path = Path("../../data/processed/historical/holdout/data.csv")
dm_holdout = LCLDataModule(
    data_path=holdout_path,
    stats_path=stats_path,
    batch_size=200,
    n_samples=20000,
)
dm_holdout.setup()

# %% [markdown]
# # 🤖 Load Pretrained Faraday model

import numpy as np
import torch

# %%
from opensynth.models.faraday.model import FaradayModel

# %%
faraday1500 = torch.load("faraday_model_1500.pt", weights_only=False)
faraday150 = torch.load("faraday_model_150.pt", weights_only=False)
faraday50 = torch.load("faraday_model_50.pt", weights_only=False)
faraday10 = torch.load("faraday_model_10.pt", weights_only=False)
# faraday1 = torch.load("faraday_model_1.pt", weights_only=False)

# %% [markdown]
# # 1️⃣ Reconstruction Attack

# %% [markdown]
# As described in the Defining good paper, the reconstruction attack involves:
#
# 1. Train the generative model with implausible outliers injected (100)
# 2. Generate a random sample of data from the injected outliers (10000)
# 3. Calculate the pairwise euclidean distances of each injected outlier with every sample in the batch of randomly generated data.
# 4. Aggregate the results to retrieve the nearest generated sample for each outlier, and express the distance as a ratio of the norm of the outlier's vector.
# 5. Visualise results in a cumulative distribution function.

# %% [markdown]
# ### 🏭 Create Reconstruction Attack Dataset

# %%
from opensynth.evaluation.privacy import reconstruction_attack

faraday1500_attack_dataset = reconstruction_attack.create_attack_dataset(
    model=faraday1500, dm=dm_with_outliers, n_samples=10000
)

faraday150_attack_dataset = reconstruction_attack.create_attack_dataset(
    model=faraday150, dm=dm_with_outliers, n_samples=10000
)

faraday50_attack_dataset = reconstruction_attack.create_attack_dataset(
    model=faraday50, dm=dm_with_outliers, n_samples=10000
)

faraday10_attack_dataset = reconstruction_attack.create_attack_dataset(
    model=faraday10, dm=dm_with_outliers, n_samples=10000
)
# faraday1_attack_dataset = reconstruction_attack.create_attack_dataset(
#     model=faraday1, dm=dm_with_outliers, n_samples=10000
# )

# %% [markdown]
# ### 🧮 Calculating pair-wise euclidean distances

# %%

faraday1500_distance_norm = reconstruction_attack.calculate_distance_norm(
    real=faraday1500_attack_dataset.outlier_samples,
    fake=faraday1500_attack_dataset.synthetic_samples,
    group_min=True,
)

faraday150_distance_norm = reconstruction_attack.calculate_distance_norm(
    real=faraday150_attack_dataset.outlier_samples,
    fake=faraday150_attack_dataset.synthetic_samples,
    group_min=True,
)

faraday50_distance_norm = reconstruction_attack.calculate_distance_norm(
    real=faraday50_attack_dataset.outlier_samples,
    fake=faraday50_attack_dataset.synthetic_samples,
    group_min=True,
)

faraday10_distance_norm = reconstruction_attack.calculate_distance_norm(
    real=faraday10_attack_dataset.outlier_samples,
    fake=faraday10_attack_dataset.synthetic_samples,
    group_min=True,
)

# faraday1_distance_norm = reconstruction_attack.calculate_distance_norm(
#     real = faraday1_attack_dataset.outlier_samples,
#     fake = faraday1_attack_dataset.synthetic_samples,
#     group_min=True
# )

# %% [markdown]
# ### 📈 Plotting cumulative distribution function

# %% [markdown]
# In the defining good paper, it is recommended that threshold ratio be set to 0.3. This is analogous to saying that for a outlier profile with daily total consumption of 30kwh, a generated profile with total consumption within 30% (21±9kwh) is sufficient to render that outlier compromised.
#
# Using this threshold, we can see that GMMs trained with 1500 clusters, about 16.5% of outliers would have rended about 53% of outliers successfully reconstructed, whilst GMM trained with only 1 cluster is very private. All synthetic data generated lies outside of 0.6 X norm of injected outliers.

# %%
import pandas as pd
import seaborn as sns

faraday1500_distance_norm["model"] = "faraday1500"
faraday150_distance_norm["model"] = "faraday150"
faraday50_distance_norm["model"] = "faraday50"
faraday10_distance_norm["model"] = "faraday10"
# faraday1_distance_norm["model"] = "faraday1"

df_reconstruction_attack_results = pd.concat(
    [
        faraday1500_distance_norm,
        faraday150_distance_norm,
        faraday50_distance_norm,
        faraday10_distance_norm,
        # faraday1_distance_norm
    ]
).reset_index(drop=True)

sns.ecdfplot(data=df_reconstruction_attack_results, x="ratio", hue="model")
plt.xlim(0, 1)

# %% [markdown]
# # 2️⃣ Membership Inference Attacks

# %% [markdown]
# As described in the Defining good paper, the membership inference attack involves:
#
# 1. Train a discriminator that attempts to distinguish between synthetically generated data and the holdout set (real data but unseen during training of the generative model.)
# 2. Create an 'attack dataset' that comprises of 1) Seen outliers during the training of the generative model 2) Unseen outliers but belonging to the same distribution to the seen outliers, and 3) Unseen outliers but belonging to a different distribution. A random guess would yield a precision of 33%.
# 3. Use the trained discriminative model to predict on the attack dataset and report on the precision, and check that it is no better than random guess.

# %% [markdown]
# ### 🏭 Create Membership Inference Attack Dataset

import pytorch_lightning as pl

# %%
from opensynth.evaluation.privacy import membership_inference_attack as mia

dm_mia_faraday1500 = mia.MembershipInferenceDataModule(
    model=faraday1500,
    dm_train=dm_with_outliers,
    dm_holdout=dm_holdout,
    batch_size=5000,
)
dm_mia_faraday1500.setup("")

dm_mia_faraday150 = mia.MembershipInferenceDataModule(
    model=faraday150,
    dm_train=dm_with_outliers,
    dm_holdout=dm_holdout,
    batch_size=5000,
)
dm_mia_faraday150.setup("")

dm_mia_faraday50 = mia.MembershipInferenceDataModule(
    model=faraday50,
    dm_train=dm_with_outliers,
    dm_holdout=dm_holdout,
    batch_size=5000,
)
dm_mia_faraday50.setup("")

# dm_mia_faraday1 = mia.MembershipInferenceDataModule(
#     model=faraday1,
#     dm_train=dm_with_outliers,
#     dm_holdout=dm_holdout,
#     batch_size=5000
# )
# dm_mia_faraday1.setup("")

# %% [markdown]
# ### 🤖 Train discriminator models

# %% [markdown]
# #### GMM with 1500 Clusters

# %%
mia_model_faraday1500 = mia.MembershipInferenceModel(
    learning_rate=0.02, input_size=48
)
train_dl = dm_mia_faraday1500.train_dataloader()
eval_dl = dm_mia_faraday1500.eval_dataloader()
trainer = pl.Trainer(max_epochs=100, accelerator="auto", strategy="auto")
trainer.fit(
    mia_model_faraday1500, train_dataloaders=train_dl, val_dataloaders=eval_dl
)

# %% [markdown]
# #### GMM with 150 Clusters

# %%
mia_model_faraday150 = mia.MembershipInferenceModel(
    learning_rate=0.02, input_size=48
)
train_dl = dm_mia_faraday150.train_dataloader()
eval_dl = dm_mia_faraday150.eval_dataloader()
trainer = pl.Trainer(max_epochs=100, accelerator="auto", strategy="auto")
trainer.fit(
    mia_model_faraday150, train_dataloaders=train_dl, val_dataloaders=eval_dl
)

# %% [markdown]
# #### GMM with 50 Clusters

# %%
mia_model_faraday50 = mia.MembershipInferenceModel(
    learning_rate=0.02, input_size=48
)
train_dl = dm_mia_faraday50.train_dataloader()
eval_dl = dm_mia_faraday50.eval_dataloader()
trainer = pl.Trainer(max_epochs=100, accelerator="auto", strategy="auto")
trainer.fit(
    mia_model_faraday50, train_dataloaders=train_dl, val_dataloaders=eval_dl
)

# %% [markdown]
# #### GMM with only 1 Cluster

# %%
# mia_model_faraday1 = mia.MembershipInferenceModel(learning_rate=0.02, input_size=48)
# train_dl = dm_mia_faraday1.train_dataloader()
# eval_dl = dm_mia_faraday1.eval_dataloader()
# trainer = pl.Trainer(max_epochs=100, accelerator="auto", strategy="auto")
# trainer.fit(mia_model_faraday1, train_dataloaders=train_dl, val_dataloaders=eval_dl)

# %% [markdown]
# ### 🎯 Check Precision

# %% [markdown]
# Results show that Faraday trained with 1500 and 150 GMM Clusters failed membership inference attacks, whilst 50 GMM Clusters and 1 GMM cluster passed. However we also see that 50 GMM cluster failed the reconstruction attack, hence it is not safe for release.
#
# In the `faraday_tutorial.ipynb` notebook, we also see that GMM with only 1 clusters is less accurate compared to 50, 150 and 1500 clusters. There is therefore a trade-off between privacy and fidelity. Perhaps there is a cluster between 1 and 50 that could meet both privacy and fidelity requirements.

# %%
df_mia_results_faraday1500 = mia.get_mia_predictions(
    mia_model_faraday1500, dm_mia_faraday1500
)
mia.print_mia_results(df_mia_results_faraday1500)

# %%
df_mia_results_faraday150 = mia.get_mia_predictions(
    mia_model_faraday150, dm_mia_faraday150
)
mia.print_mia_results(df_mia_results_faraday150)

# %%
df_mia_results_faraday50 = mia.get_mia_predictions(
    mia_model_faraday50, dm_mia_faraday50
)
mia.print_mia_results(df_mia_results_faraday50)

# %%
# df_mia_results_faraday1 = mia.get_mia_predictions(mia_model_faraday1, dm_mia_faraday1)
# mia.print_mia_results(df_mia_results_faraday1)

# %% [markdown]
# #### Check MIA attack outlier distribution

# %%
dfseen = dm_mia_faraday1500.attack_dataset.df.query("type=='seen'")
dfunseen_same = dm_mia_faraday1500.attack_dataset.df.query(
    "type=='unseen_same'"
)
dfunseen_diff = dm_mia_faraday1500.attack_dataset.df.query(
    "type=='unseen_diff'"
)
assert len(dfseen) == len(dfunseen_same) == len(dfunseen_diff) == 100

# %%
seen_tensor = np.array(dfseen["tensors"].tolist())
unseen_same_tensor = np.array(dfunseen_same["tensors"].tolist())
unseen_diff_tensor = np.array(dfunseen_diff["tensors"].tolist())

tsne_input = np.concatenate(
    [seen_tensor, unseen_same_tensor, unseen_diff_tensor]
)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
tsne_output = tsne.fit_transform(tsne_input)

# %%
plt.scatter(tsne_output[:100, 0], tsne_output[:100, 1], label="Seen")
plt.scatter(
    tsne_output[100:200, 0], tsne_output[100:200, 1], label="Unseen (Same)"
)
plt.scatter(tsne_output[200:, 0], tsne_output[200:, 1], label="Unseen (Diff)")
plt.legend()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(seen_tensor)

seen_pca = pca.transform(seen_tensor)
unseen_same_pca = pca.transform(unseen_same_tensor)
unseen_diff_pca = pca.transform(unseen_diff_tensor)

plt.scatter(seen_pca[:, 0], seen_pca[:, 1], label="Seen")
plt.scatter(
    unseen_same_pca[:, 0], unseen_same_pca[:, 1], label="Unseen (Same)"
)
plt.scatter(
    unseen_diff_pca[:, 0], unseen_diff_pca[:, 1], label="Unseen (Diff)"
)
plt.legend()

# %%

# %% [markdown]
#
