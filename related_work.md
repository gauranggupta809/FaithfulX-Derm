# Related Work & Research Gap

This document summarizes the core literature surveyed on November 15, 2026. Every paper listed below was directly opened, read, and cross-referenced.

---

## 1. Domain-Specific Studies (Skin Cancer & Medical Imaging XAI)

### 1.1 Robustness, Stability, and Fidelity of Explanations for a Deep Skin Cancer Classification Model
* **Authors**: Mirka Saarela and Lilia Georgieva
* **Year**: 2022
* **Journal**: *Applied Sciences*, 12(19), 9545
* **URL**: [https://doi.org/10.3390/app12199545](https://doi.org/10.3390/app12199545)
* **What they did**: The authors evaluated and compared Integrated Gradients (gradient-based) and LIME (perturbation-based) on a CNN trained on the 7-class HAM10000 skin lesion dataset, assessing local fidelity and quantitative stability.
* **How this project differs**: Saarela and Georgieva compared attribution algorithms against each other without calibrating against a harmless "null" baseline edit, and they did not investigate whether explanation instability under natural corruptions can detect classification errors or provide orthogonal signal beyond softmax confidence.

---

### 1.2 Assessing the (Un)Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging
* **Authors**: Nishanth Arun, Nathan Gaw, Praveer Singh, Ken Chang, Mehak Aggarwal, Bryan Chen, Katharina Hoebel, Sharut Gupta, Jay Patel, Mishka Gidwani, Julius Adebayo, Matthew D. Li, and Jayashree Kalpathy-Cramer
* **Year**: 2021 (arXiv preprint 2020)
* **Journal**: *Radiology: Artificial Intelligence*, 3(6), e200267
* **URL**: [https://arxiv.org/abs/2008.02766](https://arxiv.org/abs/2008.02766) / [https://doi.org/10.1148/ryai.2021200267](https://doi.org/10.1148/ryai.2021200267)
* **What they did**: The authors systematically benchmarked eight saliency map techniques across chest radiography datasets, evaluating them against four criteria: localization utility against ground-truth segmentation masks, model parameter randomization, repeatability under subtle perturbations, and run-to-run reproducibility.
* **How this project differs**: Arun et al. evaluated whether saliency maps can serve as reliable spatial disease localizers compared to dedicated segmentation networks; in contrast, our study asks whether the *instability* of explanation maps under minor synthetic perturbations can be repurposed as a statistical warning signal to detect model misclassification.

---

### 1.3 Safer Skin Lesion Classification with Global Class Activation Probability Map Evaluation and SafeML
* **Authors**: Kuniko Paxton, Koorosh Aslansefat, Amila Akagić, Dhavalkumar Thakker, and Yiannis Papadopoulos
* **Year**: 2025
* **Preprint**: arXiv:2508.20776 [cs.CV]
* **URL**: [https://arxiv.org/abs/2508.20776](https://arxiv.org/abs/2508.20776)
* **What they did**: Proposed Global Class Activation Probabilistic Map Evaluation (GCAPM) combined with SafeML distance metrics to perform runtime monitoring of skin cancer classifiers, flagging potential false diagnoses when activation distributions diverge.
* **How this project differs**: Paxton et al. designed an all-class probabilistic activation map coupled with statistical distance monitoring across latent representations; our project evaluates standard, widely deployed local attribution maps (Grad-CAM, Integrated Gradients) under calibrated physical perturbations and directly tests their added diagnostic value over model confidence in multi-class dermoscopy.

---

### 1.4 Revisiting the Trustworthiness of Saliency Methods in Radiology AI
* **Authors**: Jiajin Zhang, Hanqing Chao, Giridhar Dasegowda, Ge Wang, Mannudeep K. Kalra, and Pingkun Yan
* **Year**: 2024
* **Journal**: *Radiology: Artificial Intelligence*, 6(1), e220221
* **URL**: [https://pmc.ncbi.nlm.nih.gov/articles/PMC10831523/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10831523/) / [https://doi.org/10.1148/ryai.220221](https://doi.org/10.1148/ryai.220221)
* **What they did**: Evaluated saliency methods on CheXpert chest X-rays and brain MRIs under subtle input perturbations using prediction-saliency correlation (PSC) and radiologist reader studies, finding that saliency maps are brittle and often uncorrelated with prediction degradation.
* **How this project differs**: Zhang et al. focused on demonstrating that radiologists are easily misled by brittle heatmaps when predictions degrade; our work in dermatology asks whether that brittleness/instability can be formally calibrated against null baselines and used systematically as an automated statistical detector of incorrect model predictions.

---

## 2. Foundational XAI & Robustness Literature

### 2.1 Interpretation of Neural Networks is Fragile
* **Authors**: Amirata Ghorbani, Abubakar Abid, and James Zou
* **Year**: 2019
* **Conference**: *AAAI Conference on Artificial Intelligence* (AAAI-19)
* **URL**: [https://arxiv.org/abs/1710.10547](https://arxiv.org/abs/1710.10547)
* **What they did**: Demonstrated that gradient-based feature importance maps (including Integrated Gradients and DeepLIFT) can be altered drastically by imperceptible adversarial perturbations that leave the model's predicted class label unchanged.
* **How this project differs**: Ghorbani et al. investigated adversarial worst-case vulnerability; our study evaluates realistic, non-adversarial imaging shifts (blur, brightness, noise, JPEG compression), establishes a non-zero "null" baseline to measure excess instability, and investigates whether natural instability carries diagnostic utility to flag classification errors.

---

### 2.2 Sanity Checks for Saliency Maps
* **Authors**: Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim
* **Year**: 2018
* **Conference**: *Neural Information Processing Systems* (NeurIPS 2018)
* **URL**: [https://arxiv.org/abs/1810.03292](https://arxiv.org/abs/1810.03292)
* **What they did**: Introduced model parameter randomization and label randomization tests, demonstrating that methods like Guided Backpropagation act as edge detectors independent of network weights, whereas Grad-CAM and Integrated Gradients pass weight randomization checks.
* **How this project differs**: Adebayo et al. established whether saliency methods are fundamentally coupled to network weights; our study uses the two methods that pass their sanity check (Grad-CAM and IG) and evaluates their stability under realistic input domain shifts in medical imaging.

---

### 2.3 A Benchmark for Interpretability Methods
* **Authors**: Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim
* **Year**: 2019
* **Conference**: *Neural Information Processing Systems* (NeurIPS 2019)
* **URL**: [https://arxiv.org/abs/1806.10758](https://arxiv.org/abs/1806.10758)
* **What they did**: Proposed the RemOve And Retrain (ROAR) framework to measure explanation faithfulness by retraining models from scratch on images where the most attributed pixels were removed.
* **How this project differs**: ROAR retrains models repeatedly, which is computationally prohibitive for post-hoc clinical pipelines; our work evaluates test-time post-hoc deletion/insertion benchmarked directly against an explicit random-ordered cell deletion control, combined with Gaussian blur masking to minimize out-of-distribution distribution shift.

---

### 2.4 RISE: Randomized Input Sampling for Explanation of Black-box Models
* **Authors**: Vitali Petsiuk, Abir Das, and Kate Saenko
* **Year**: 2018
* **Conference**: *British Machine Vision Conference* (BMVC 2018)
* **URL**: [https://arxiv.org/abs/1806.07404](https://arxiv.org/abs/1806.07404)
* **What they did**: Formulated the deletion (removal) and insertion (addition) curve metrics, evaluating explanations by measuring Area Under the Curve (AUC) when masking pixels in order of attributed importance.
* **How this project differs**: Petsiuk et al. introduced AUC deletion/insertion without a standardized null-hypothesis baseline; our project benchmarks attribution-ordered curves directly against an explicit random-ordered cell deletion control and evaluates whether deletion/insertion gap correlates with perturbation stability.

---

## 3. Research Gap Statement

Among the 8 sources we checked on November 15, 2026, we did not find a study that (a) calibrates explanation instability against a null perturbation, and (b) tests whether it adds information beyond model confidence for detecting errors in 7-class dermoscopy. This does not establish that no such study exists.
