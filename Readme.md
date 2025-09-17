# A Multi-Indicator Coupled Spatio-Temporal Latent Factorization of Tensors Model for Signal Recovery in Wireless Sensor Networks



## 📌 Introduction

Wireless Sensor Networks (WSNs) have gained increasing attention as a key enabler of intelligent sensing across diverse applications. However, in practical deployments, WSNs often encounter missing data caused by various internal or external factors, such as sudden sensor failures or intentional sabotage.

Low-rank matrix approximation (LRMA) methods are widely adopted to recover missing data in WSNs. Nevertheless, existing LRMA-based approaches focus on modeling single indicators, neglecting the inherent correlations among multiple indicators — thereby limiting recovery accuracy.

To address this issue, this paper proposes a **M**ulti-**I**ndicator coupled **S**patio-**T**emporal **L**atent **F**actorization of **T**ensors (**MIST-LFT**) model. Its core innovations are:

1. A tensor latent factorization framework that **couples multi-indicator data**, incorporating a learnable weighted *L₂-norm* to adaptively balance each indicator’s contribution during training.
2. Explicit modeling of **spatio-temporal correlations** via regularization constraints, enabling the model to capture both spatial topology and temporal dynamics of WSNs.

As a result, MIST-LFT effectively exploits **multi-indicator coupling** and **spatio-temporal structure**, achieving significantly higher recovery accuracy — especially under high missing rates.

Extensive experiments on two real-world WSN datasets demonstrate that MIST-LFT consistently outperforms six state-of-the-art baselines.

---

## ⚙️ Experimental environment

- **Java Development Kit (JDK)**: 1.8（**Need**）  

