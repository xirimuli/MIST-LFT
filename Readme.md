# A Spatiotemporally-Regularized Latent Factorization of Tensors Model for Representation Learning of Nonstandard Environmental Sensing Data

Wireless Sensor Networks (WSNs) are an essential component of intelligent environmental monitoring systems. However, the sensing data collected by WSNs are often **nonstandard** due to issues such as missing values, sudden sensor failures, or wireless signal dropouts.  

To tackle this challenge, we propose **SR-LFT (Spatiotemporally-Regularized Latent Factorization of Tensors)**, a novel representation learning model designed for nonstandard environmental sensing data.

---

## 🚀 Key Ideas
1. **Latent Factorization of Tensors (LFT) Framework**  
   - Jointly models multiple sensing indicators.  
   - Introduces a *learnable indicator-loss weight* to adaptively balance the contribution of each indicator during training.  

2. **Spatiotemporal Regularization**  
   - Explicitly incorporates both **spatial structures** and **temporal patterns** as regularization constraints.  
   - Enhances the model’s ability to leverage dependencies across multiple sensing indicators.  

---

## ✨ Contributions
- A new tensor-based latent factorization model with spatiotemporal regularization.  
- Adaptive weighting mechanism for heterogeneous sensing indicators.  
- Significant improvements in representation learning for nonstandard WSN data.  

---

## 📊 Experimental Results
We evaluate SR-LFT on **two real-world WSN datasets** with high missing rates.  
- SR-LFT consistently and significantly outperforms **six state-of-the-art baselines**.  
- Demonstrates strong robustness in handling missing values while preserving spatiotemporal correlations.  

---

## ⚙️ Experimental environment

- **Java Development Kit (JDK)**: 1.8（**Need**）  

