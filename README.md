# Semi-Negative Contrastive Subclass Discriminative Network for Compositional Zero-shot Learning

<img width="2382" height="959" alt="network" src="https://github.com/user-attachments/assets/8065a59a-ca91-4e5a-9b55-f6839124d033" />

Semi-Negative Contrastive Subclass Discriminative
Network for Compositional Zero-shot Learning
Yang Liu, Xinshuo Wang, Xinbo Gao, Jungong Han, Ling Shao

# Abstract
The goal of compositional zero-shot learning (CZSL) is to train a model to recognize images containing known attributeobject pairs, thus reducing the reliance on extensive training data and enabling the model to identify unseen combinations. Current CZSL methods face several challenges, including multiple attributes for a single object, disconnected training and test sets, long-tailed distribution of visual categories, and substantial differences in state representation between different objects. These factors collectively impede the precise identification of new combinations. In response to these challenges, we propose a SemiNegative Contrastive Subclass Discriminative Network (SN-CSDN) based on contrastive learning. Firstly, we propose a semi-negative sampling strategy that incorporates carefully selected negative samples into the training process. This approach enables the model to effectively distinguish between different classes while enhancing
its ability to capture fine-grained subclass features. By improving the model’s sensitivity to inter-class differences and refining its recognition of subtle intra-class variations, this strategy significantly boosts overall discrimination performance. Additionally, we introduce a decoupled network branch designed to capture the intricate relationships between attributes and objects by generating more representative compositional embeddings. This branch leverages subclass information to ensure an accurate classification of synthesized embeddings while preserving the inherent visual distinctions of the original decoupled embeddings across different combinations. By improving feature representation capacity and mitigating sample imbalance, this design effectively improves model performance in long-tailed distributions. Our method has
been comprehensively evaluated on three benchmark datasets, with results showing significant performance improvements that demonstrate the method’s effectiveness and reliability.
