# Linguistic Signatures in E-Commerce: Evaluating Perplexity and Burstiness for the Detection of LLM-Generated Amazon Reviews

**Group 5:** Sindusara Munasinghe, Adrian Di Paola, Raelen Garces, and Simon Liu

## Abstract
The rapid growth of Large Language Models (LLMs) has presented a systemic threat to the integrity of online e-commerce platforms by enabling the mass generation of contextually accurate synthetic reviews. This project presents a data-driven methodology to distinguish authentic, human-written Amazon reviews from Al-generated text through deep linguistic analysis. Utilizing the Amazon Reviews 2023 dataset, we constructed a controlled, matched-pair dataset of 2,000 reviews, pairing verified human purchases with contextually identical synthetic counterparts generated via LLaMA-3.1-8B-Instant. By extracting key statistical linguistic signatures, specifically perplexity (text predictability) and burstiness (structural rhythm), we established foundational metrics for evaluating synthetic text. Our evaluation progressed from statistical distribution validation to machine learning classification. While initial tests confirmed statistically significant differences between the writing styles, the feature spaces showed that there was dense overlap. Due to this, baseline linear models (Logistic Regression) struggled to accurately separate the classes, plateauing at 74.49% accuracy and frequently misclassifying human writing. To address this non-linear separability, we deployed advanced non-linear architectures. By dynamically segmenting the feature space, XGBoost emerged as the superior model, capturing complex stylistic boundaries to achieve 81.82% accuracy and drastically improving human text recall to 0.84. Ultimately, this research validates intrinsic linguistic analysis as a viable baseline for detection and underscores the necessity of adaptive, non-linear algorithms in the ongoing arms race against sophisticated generative Al.

**Index Terms** — E-Commerce, Al Text Detection, Natural Language Processing, Large Language Models, Perplexity, Burstiness, Machine Learning Classification, XGBoost.

---

## 1. INTRODUCTION

The rapid advancement of Large Language Models (LLMs) has introduced significant challenges to the integrity of online e-commerce platforms. On platforms like Amazon, consumer trust and purchasing decisions rely heavily on the authenticity of user reviews. However, LLMs have made it increasingly easy for malicious groups to generate high-quality, contextually accurate fake reviews at scale, bypassing traditional detection systems that previously could flag poor grammar and repetitive phrasing. This project aims to address this challenge by researching and developing a data-driven methodology to distinguish human-written Amazon reviews from Al-generated ones using deep linguistic analysis methods. 

To provide a controlled comparison, we utilize the Amazon Reviews 2023 dataset as our source of ground-truth human writing, generating a synthetic counterpart dataset using a modern LLM. We focus purely on textual signatures rather than any metadata information, targeting the problem of detecting fake reviews using only the text itself. The main groups that benefit from a successful outcome of this project are e-commerce platforms seeking to improve review moderation, consumers relying on authentic reviews, and sellers who are impacted by manufactured review-farm activity. 

The main contributions of this research are:
* **Matched-Pair Dataset Creation:** The construction of a balanced dataset that isolates linguistic style from content by ensuring human and Al reviews describe the exact same product, rating, and word length.
* **Statistical Validation of Linguistic Features:** A detailed evaluation of how individual statistical metrics, being structural variance (burstiness) and text predictability (perplexity), function as independent indicators of LLM-written text in the specific commercial context.
* **Model Evaluation:** A comparative analysis of linear and non-linear classification models, demonstrating how advanced architectures capture complex stylistic patterns to effectively separate Al-generated reviews from genuine human feedback.

## 2. RELATED WORK

### 2.1 Linguistic and Statistical Signatures
**[4] Xu and Sheng (2024):** This study introduces "Targeted Masking Perturbation" to detect Al-generated content by measuring the predictability (perplexity) and structural variance (burstiness) of text. Their mathematical definitions of these metrics provide the foundational framework for our RQ1 and RQ2, which aim to validate if these same signatures remain effective when applied to Amazon reviews.

**[1] Mudasir et al. (2024):** The authors propose a detection framework using deep learning architectures like BiLSTM and CNNs to identify Al-generated spam. This relates to our RQ3 by providing a performance benchmark for how multi-feature predictive models can distinguish synthetic reviews from genuine ones.

### 2.2 Dataset and Contextual Baselines
**[2] Hou et al. (2024):** This research introduces the Amazon Reviews 2023 dataset, a massive corpus of 570 million reviews used to bridge natural language with item metadata. We utilize this specific dataset to provide the "Human-Written" ground truth for our comparative analysis against LLM-generated text.

**[3] Xie et al. (2012):** This work identifies review spam by detecting abnormal temporal bursts in posting history. While our current study pivots away from temporal metadata due to the constraints of synthetic data generation, this work provides the necessary context for why our project focuses instead on the "singleton" problem-where bots with no history must be caught using linguistic signatures alone.

## 3. METHODOLOGY

To address our research questions, the project methodology was structured into three sequential phases: dataset generation, feature extraction, and model classification.

### 3.1 Dataset Generation
A significant challenge in detecting Al-generated text in the wild is the lack of pre-labeled ground truth. The utilized Amazon Review 2023 dataset does not include any strong indicator on if a review is Al generated or not. To overcome this, we constructed a balanced, matched-pair dataset of 2,000 reviews. We first filtered the Electronics category of the Amazon Reviews 2023 dataset to extract 1,000 human-written reviews. To ensure human written reviews, we took reviews marked strictly with the "Verified Purchase" flag. 

For the synthetic counterpart, we utilized the llama-3.1-8b-instant model. Through targeted prompting, the LLM generated 1,000 synthetic reviews designed to match the specific product context, star rating, and word count of their human-written counterparts. This matched-pair design allows us to isolate linguistic style from contextual subject matter.

### 3.2 Feature Extraction
Once the dataset was constructed, we extracted two primary statistical linguistic signatures for both classes:
* **Perplexity:** Calculated using the pre-trained GPT-2 model, perplexity measures the statistical predictability of a sequence of words. Lower perplexity indicates highly predictable text (common shown by LLMs, which predict the most probable next token), whereas higher perplexity shows the natural randomness and creative vocabulary of human writers.
* **Burstiness:** Calculated as the standard deviation of sentence lengths within a review. This metric captures structural variance. Humans tend to write with varied rhythms, mixing short and long sentences, resulting in higher burstiness, while LLMs tend to maintain a uniform, average sentence length.

To evaluate the actual statistical significance of these individual metrics between the human and Al classes, we performed both the Mann-Whitney U test and the Welch's T-test. Furthermore, we utilized a Multivariate Analysis of Variance (MANOVA) to assess the statistical significance of the combined feature set.

### 3.3 Model Classification Pipeline
The dataset was split into an 80% training set and a 20% held-out test set. We developed models in two stages. First, we established a linear baseline using Logistic Regression to evaluate the individual and combined predictive power of perplexity and burstiness. Specifically, we trained independent linear models using only burstiness and only perplexity to isolate their individual predictive strengths, before training a third linear model that combined both features. 

Following this, we implemented non-linear architectures, specifically Random Forest and XGBoost classifiers. Unlike the linear baseline, these non-linear models were exclusively trained using the combined feature set. These models were selected to determine if non-linear methods could capture the complex boundaries in the feature space where human and Al distributions heavily overlap.

## 4. DATASET

The primary dataset used for this study is the large-scale Amazon Reviews dataset collected in 2023 by McAuley Lab. Due to the massive scale of the corpus, which exceeds 570 million reviews, we constrained our scope specifically to the Electronics category. The raw data provides a comprehensive view of e-commerce interactions, encompassing User Reviews, Item Metadata, and Links.

For our human-written ground truth, we utilized specific user review attributes including text, rating, and the `verified_purchase` flag. We filtered exclusively for reviews where `verified_purchase` was true, establishing a high-confidence baseline for authentic human activity and reducing the likelihood of including existing bot-generated content in our "human" class.

To create a balanced and controlled comparison, we generated a synthetic counterpart for each human review through a matched-pair system. For every human review in our final selection, we extracted four key parameters: the product's Amazon Standard Identification Number (ASIN) to identify the item, the item title so the model could understand what it was writing the review for, the specific star rating (1-5), and the total word count. We then utilized the llama-3.1-8b-instant model to generate a new review for that same product and rating. 

The LLM was provided with the product metadata and specific instructions to match the approximate length of its human counterpart. This methodology was necessary because it isolates linguistic style-such as word choice and sentence rhythm-from the content or length of the review. By ensuring both the human and Al review describe the exact same product with the same sentiment and length, we prevent classification models from relying on context differences, forcing them to distinguish between classes based purely on deep linguistic signatures.

The resulting curated dataset consists of 2,002 total instances, balanced perfectly between 1,001 human reviews and 1,001 LLM-generated reviews. Table 1 outlines the final statistics of the dataset.

**TABLE 1**
Basic Statistics of the Matched-Pair Dataset

| Metric | Human-Written | LLM-Generated |
| :--- | :--- | :--- |
| Total Samples | 1,001 | 1,001 |
| Average Word Count | 51.0 | 60.3 |
| Average Sentence Count | 4.4 | 4.0 |
| Mean Rating | 4.3 | 4.3 |

## 5. EXPERIMENTS AND RESULTS

### 5.1 RQ1: Statistical Significance of Linguistic Features

Our first research question aimed to validate whether statistically significant differences exist in the distributions of perplexity and burstiness between human and Al-generated reviews. We structured this analysis in three phases: visual distribution analysis, independent feature testing, and combined multivariate analysis.

#### 5.1.1 Visual Distribution Analysis
Visual analysis of the feature distributions provides strong preliminary evidence of distinct writing signatures.

![Fig. 1. Perplexity distributions for Human vs. Al reviews. View A shows the standard scale with outliers removed, while View B highlights the logarithmic separation.](outputs/fig_1.png)

**Perplexity Analysis:** The perplexity distributions reveal a clear "two-hill" separation, most visible in the logarithmic scale (Fig. 1, View B). Al-generated reviews (red) are heavily clustered at the lower end of the "Weirdness Score," representing the high predictability inherent in LLM token selection. In contrast, human reviews (blue) exhibit a significantly wider and higher distribution, showcasing the natural randomness of organic writing.

![Fig. 2. Burstiness distributions for Human vs. Al reviews. View A displays the structural variance on a standard scale, while View B shows the log-scale isolating the non-zero distribution.](outputs/fig_2.png)

**Burstiness Analysis:** The burstiness distributions (Fig. 2) highlight a structural disparity in sentence construction. Al reviews demonstrate a sharp spike near zero on the standard scale (View A), indicating highly uniform sentence lengths. Human writing, however, shows much higher structural variance and rhythmic unpredictability, with the log-scale peak shifted further to the right.

#### 5.1.2 Independent Feature Testing
To mathematically confirm these visual observations, we first conducted independent statistical tests for each feature, as detailed in Table 2. Both Welch's T-test and the Mann-Whitney U test yielded p-values well below the 0.05 threshold, allowing us to decisively reject the null hypothesis for both individual features.

**TABLE 2**
Independent Feature Statistical Significance

| Feature | Welch's T-Test (p-value) | Mann-Whitney U (p-value) |
| :--- | :--- | :--- |
| Perplexity | $1.04\times10^{-3}$ | $1.91\times10^{-123}$ |
| Burstiness | $1.14\times10^{-17}$ | $8.31\times10^{-11}$ |

Beyond simply proving that a difference exists, we quantified the magnitude of these differences using descriptive statistics and effect sizes (Table 3). 

**TABLE 3**
Detailed Statistical Comparison of Linguistic Features

| Feature | Human ( $\sigma$) | $AI(\mu\pm\sigma)$ | Test Stat | Effect Size |
| :--- | :--- | :--- | :--- | :--- |
| Perplexity | 530.954620.63 | $\googlelongdiv{15.34+52.36}$ | $U=789318.0$ | $r=-0.612!$ |
| Burstiness | $4.47\pm5.18$ | $2.87\pm2.63$ | $t=8.6696$ | $d=0.3922$ |

The descriptive statistics highlight a clear behavioral difference between the two classes. Standard deviation ($\sigma$) measures the amount of variation or "spread" in a dataset. For perplexity, human writing shows massive variation ($\sigma=4620.63$), reflecting a wide, creative, and unpredictable use of vocabulary. In contrast, Al text is highly constrained and formulaic ($\sigma=52.36$). A similar trend appears in burstiness, where human writing exhibits nearly double the standard deviation (Human $\sigma=5.18$ vs. Al $\sigma=2.63$) indicating that human writers naturally mix long and short sentences far more than Al models do.

#### 5.1.3 Combined Multivariate Analysis (MANOVA)
To evaluate the features while combined, we conducted a Multivariate Analysis of Variance (MANOVA). The results for the classification label across all four multivariate criteria are summarized in Table 4. The tests returned p-values of $<0.0001$ with an $F(2,1976)=46.1036$.

**TABLE 4**
Multivariate Analysis of Variance (MANOVA) for the Label Effect

| Test Statistic | Value | F-Value | DF (Num, Den) | p-value |
| :--- | :--- | :--- | :--- | :--- |
| Wilks' lambda | 0.9554 | 46.1036 | (2, 1976) | <0.0001 |
| Pillai's trace | 0.0446 | 46.1036 | (2, 1976) | <0.0001 |
| Hotelling-Lawley trace | 0.0467 | 46.1036 | (2, 1976) | <0.0001 |
| Roy's greatest root | 0.0467 | 46.1036 | (2, 1976) | <0.0001 |

The Wilks' Lambda value of 0.9554 indicates that these two features combined explain approximately 4.46% of the total variance between the classes. This confirms that the combination of perplexity and burstiness provides a statistically robust foundation for distinguishing between human and Al text. However, the relatively low explained variance also mathematically shows the dense overlap in the feature space, justifying the progression to the advanced machine learning classification models planned in Research Questions 2 and 3.

### 5.2 RQ2: Linear Classification Performance

To assess the baseline predictive capabilities of our extracted linguistic signatures, we developed a classification pipeline using Logistic Regression. While RQ1 established that statistical differences exist between Human and Al text distributions, RQ2 investigates whether these classes are linearly separable. By evaluating linear models first, we establish a foundational baseline before progressing to complex architectures. Model performance was evaluated on a 20% held-out test set using Accuracy, Recall, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

#### 5.2.1 Independent Feature Predictive Power
We first trained independent models to isolate the diagnostic strength of each feature. As detailed in Table 5, utilizing Burstiness as a standalone feature resulted in an accuracy of 51.01% and an ROC-AUC of 0.5541, rendering it statistically indistinguishable from random chance. This aligns with our distribution findings in RQ1. While the variances differ, the dense clustering of both human and Al sentence lengths near the distribution mean makes it difficult for a single linear threshold to accurately divide the classes.

Perplexity demonstrated better but still moderate independent predictive power, achieving 66.16% accuracy and an ROC-AUC of 0.7960. The confusion matrix analysis for this model revealed a strong bias toward minimizing false positives for Al (achieving an F1-Score of 0.71), but it struggled significantly with recall for genuine human reviews (0.48), misclassifying highly structured human writing as synthetic.

**TABLE 5**
Logistic Regression Performance Metrics (Test Split)

| Features Used | Acc. | ROC-AUC | F1 (AI) | Rec. (H) |
| :--- | :--- | :--- | :--- | :--- |
| Perplexity Only | 66.16% | 0.7960 | 0.71 | 0.48 |
| Burstiness Only | 51.01% | 0.5541 | 0.54 | 0.45 |
| Combined (Both) | 74.49% | 0.8055 | 0.77 | 0.65 |

#### 5.2.2 Synergistic Combined Performance
Next, we trained a unified logistic regression model incorporating both linguistic features. 

![Fig. 3. Confusion Matrices demonstrating the shift in classification errors across the Perplexity-only, Burstiness-only, and Combined linear models.](outputs/fig_3.png)

The integration of both metrics showed a significant performance increase. The combined model achieved an accuracy of 74.49% and an ROC-AUC of 0.8055. This indicates a synergistic relationship: while Burstiness is an ineffective separator in a one-dimensional space, it provides critical contextual variance when mapped alongside Perplexity in a two-dimensional feature space.

![Fig. 4. Linear Decision Boundary of the combined Logistic Regression model.](outputs/fig_4.png)

However, as visualized in Figure 4, the linear decision boundary shows the inherent limitations of this approach. There remains a dense, overlapping region where authentic human reviews and Al-generated reviews have the same linguistic footprint. Al reviews are tightly clustered in the low-perplexity, low-burstiness quadrant. Human reviews display much wider variance, but many data points still cross the linear threshold into the "predictable" zone. This overlap restricts the human recall to 0.65, confirming that while linguistic signatures are valid indicators, their relationship is highly non-linear, justifying the need for the advanced non-linear machine learning techniques explored in RQ3.

### 5.3 RQ3: Non-Linear Model Performance

The findings in RQ2 demonstrated a critical limitation of linear classification: while clear statistical differences exist between human and synthetic text, the feature spaces heavily overlap. Specifically, Logistic Regression struggled to identify genuine human reviews that happened to be written with high predictability, resulting in a low human recall of 0.65. To capture these multidimensional stylistic nuances, we evaluated two non-linear ensemble architectures: Random Forest (a bagging approach) and XGBoost (a gradient boosting approach).

#### 5.3.1 Ensemble Classification Results
Both non-linear models were trained exclusively on the combined feature set (Perplexity and Burstiness) and evaluated against the same 20% test set to ensure a direct comparison with the linear baseline.

**TABLE 6**
Comparative Classification Metrics (Test Split)

| Model Architecture | Acc. | ROC-AUC | F1 (AI) | Rec. (H) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | 74.49% | 0.8055 | 0.77 | 0.65 |
| Random Forest | 81.57% | 0.8929 | 0.81 | 0.84 |
| XGBoost | 81.82% | 0.8982 | 0.82 | 0.84 |

As detailed in Table 6, the introduction of tree-based algorithms significantly outperformed the linear baseline. XGBoost emerged as the superior model, achieving an accuracy of 81.82% and an ROC-AUC of 0.8982. Most importantly, both non-linear models successfully mitigated the primary failure point of the linear classifier by drastically improving the recall for authentic human text (rising to 0.84). This indicates the non-linear models successfully learned to distinguish naturally predictable human writers from the LLaMA-3.1-8B generated text without utilizing a rigid threshold.

![Fig. 5. Confusion Matrices for Random Forest (Left) and XGBoost (Right), demonstrating the significant reduction in misclassified human reviews compared to the linear baseline.](outputs/fig_5.png)

#### 5.3.2 Decision Boundary Topography
The fundamental advantage of these ensemble models is visualized in their decision boundaries. Rather than bisecting the feature space, these architectures dynamically segment the data into granular, localized regions.

![Fig. 6. Non-linear Decision Boundaries (Log Scale) for Random Forest (Left) and XGBoost (Right). Both models successfully isolate clusters of authentic human writing within the dense predictable zone.](outputs/fig_6.png)

As shown in Figure 6, the models established distinct decision boundaries to separate the dense cluster of Al-generated text from the variance of human writing. The Random Forest model established a rectangular threshold, utilizing a combination of perplexity and burstiness to isolate the highly predictable synthetic text. Conversely, the XGBoost visualization reveals a predominantly vertical decision boundary. This indicates that within this two-dimensional feature space, the gradient boosting mechanism heavily prioritized perplexity as the primary discriminative signature, creating a smooth zone across the logarithmic scale to achieve its superior classification accuracy.

#### 5.3.3 Model Training Diagnostics
To ensure the quality of these advanced architectures and make sure that they did not memorize the matched-pair dataset, we analyzed their internal training dynamics and feature weighting.

![Fig. 7. Feature Importance Rankings for Random Forest (Left) and Training Loss Curve for XGBoost (Right).](outputs/fig_7.png)

The feature importance analysis (Figure 7, Left) provides an ranking of our extracted metrics. Perplexity was used as the dominant driver of classification, confirming that statistical predictability is the most distinctive signature of LLM generation. However, Burstiness also had significant importance, acting as a necessary secondary feature to analyze the structural rhythm of edge cases. Furthermore, the training loss curve for XGBoost (Figure 7, Right) demonstrates steady convergence. While the training loss continues to decrease, the validation loss plateaus and stabilizes without diverging upward. This confirms that the model successfully generalized the linguistic boundaries of the human and synthetic classes without over-fitting to the specific LLAMA-3.1 training samples.

### 5.4 Threats to Validity
Several factors may impact the validity and generalization of our findings. First, our synthetic dataset was generated exclusively using LLaMA-3.1-8B-Instant. Because this is a relatively small, highly optimized open-weight model, the linguistic signatures our classifiers identified may overfit to its specific architecture. More advanced cloud models with vastly larger parameter counts, such as OpenAI's GPT-5 or Anthropic's Claude 4.6, have greater stylistic capability and may generate text that more closely mimics organic human writing, making them potentially harder to detect using these baseline metrics.

Second, our reliance on the pre-trained GPT-2 model to calculate perplexity introduces a measurement constraint. GPT-2 utilizes an older tokenization scheme and architectural framework. Upgrading the perplexity evaluation pipeline to utilize a more modern base model with an advanced tokenizer could provide a better predictability metric, capturing subtler synthetic patterns. Furthermore, despite filtering for "Verified Purchases," the Amazon 2023 dataset may still contain undetected and realistic Al-generated reviews within our human ground truth, introducing noise into the training data.

## 6. DISCUSSION

### 6.1 Proof of Concept and The Viability of Linguistic Detection
Ultimately, this study serves as a foundational proof-of-concept for the viability of deep linguistic analysis in e-commerce moderation. By isolating the text from its metadata, the experiment successfully validates that at a fundamental level, mathematical differences in structural rhythm and text predictability do exist between organic and synthetic writing. While simplistic linear thresholds fail to distinguish the dense overlap between human and Al feature spaces, advanced, non-linear architectures like XGBoost prove that these complex stylistic boundaries can be successfully captured and mapped. This confirms that relying on intrinsic linguistic signatures is a feasible and effective baseline for modern Al detection.

### 6.2 The Generative Arms Race: Model Evolution vs. Detection Resolution
However, detecting synthetic text is not a static or simple problem, but rather an ongoing "arms race" between generative capabilities and detection resolution. Our synthetic dataset was generated exclusively using LLaMA-3.1-8B-Instant, a highly optimized but relatively small open-weight model. As the generative Al landscape rapidly evolves, large cloud models with much larger parameter counts (such as OpenAl's GPT-5 or Anthropic's Claude 4.6) are showcasing increasingly sophisticated and realitic stylistic capabilities. It is possible that future iterations of these models could be trained to introduce artificial "burstiness" and reduce the algorithmic predictability of their token selection, effectively making it more difficult to find the mathematical gap between human and synthetic distributions.

Conversely, the methodologies used to detect these models must also scale in complexity. For instance, our reliance on the pre-trained GPT-2 model to calculate perplexity introduces an inherent measurement constraint. Because GPT-2 utilizes an older tokenization scheme and architectural framework, it may lack the sensitivity required to evaluate text generated by more modern transformers. Upgrading the perplexity evaluation pipeline to utilize a more modern, billion-parameter base model with an advanced tokenizer could provide a much higher-resolution predictability metric, capturing the subtler synthetic patterns that currently slip into the overlapping feature space.

Therefore, while LLMs will continue to become more "human-like" in their cadence, detection frameworks will simultaneously become more advanced as well. Our findings suggest that the future of review moderation will not rely on a single, permanent algorithm, but rather on adaptive models that continuously update their linguistic baselines to match the shifting linguistic signatures of artificial intelligence.

### 6.3 The Broader Impact on E-Commerce Integrity
Beyond the technical challenges of detection, the spread of Al-generated reviews is a systemic threat to the e-commerce ecosystem. As large language models make it increasingly easier to make high-quality, contextually accurate feedback at an rapid scale, the internet is facing a growing flood of automated text. This synthetic content directly undermines the credibility of online review systems, which serve as the foundational trust mechanism for digital consumerism. If consumers can no longer reliably tell the difference between genuine human feedback and manufactured botted reviews, the value of e-commerce reviews will inevitably collapse. Therefore, developing modern and continuously evolving detection mechanisms is a necessary defense to preserve the integrity, transparency, and trustworthiness of online platforms in our increasingly synthetic digital age.

## 7. CONCLUSION AND FUTURE WORK

This project successfully demonstrated that deep linguistic signatures, specifically perplexity and burstiness, can effectively distinguish human-written Amazon reviews from LLM-generated ones when evaluated through the appropriate algorithms. Our statistical analysis confirmed that while foundational differences in text predictability and structural variance exist, their feature spaces heavily overlap. This makes it difficult for baseline linear models, like Logistic Regression, to accurately separate the classes, often misclassifying predictable human writing. However, by deploying advanced non-linear architectures, specifically XGBoost, the models successfully captured these complex, multidimensional stylistic boundaries, reducing the overlap and achieving effective classification accuracy.

For future work, this methodology should be expanded across multiple dimensions to rigorously test its effectiveness. First, future studies should evaluate other online domains by expanding beyond the Electronics category to other e-commerce sectors. Second, researchers should experiment with a broader range of modern and larger generative models (such as OpenAI's GPT-5 or Anthropic's Claude 4.6) to assess if detection holds against architectures with more advanced stylistic capabilities. Third, the feature extraction pipeline should be upgraded by utilizing modern, high-parameter base models with advanced tokenizers for perplexity calculation, rather than relying on the older GPT-2 architecture. 

Additionally, future experiments should explore adversarial prompting, specifically instructing LLMs to intentionally write with high burstiness and varied predictability to mimic human randomness. This helps evaluate these linguistic signatures against prompting evasion tactics. Finally, while this study successfully isolated linguistic features, reintegrating temporal and behavioral metadata, such as posting time, into a multi-level model could provide another distinguishable feature in identifying coordinated bot-farm activity in modern e-commerce.

## REFERENCES

[1] E. M. A. W. Mudasir and A. S. Kashish. Ai-generated spam review detection framework with deep learning algorithms and natural language processing. computers, 2024.

[2] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024.

[3] W. G. L. S. Xie, S. and P. S. Yu. Review spam detection via time series pattern discovery. Proceedings of the 21st International Conference on World Wide Web.635-63, 2012.

[4] Z. Xu and V. S. Sheng. Detecting ai-generated code assignments using perplexity of large language models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(21), 22285-22292, 2024.
