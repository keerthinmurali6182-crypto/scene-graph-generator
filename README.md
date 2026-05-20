# Visual Relationship Detection & Scene Graph Representation Engine

This repository hosts a multi-stage **Scene Graph Generation (SGG) pipeline** built with PyTorch. It bridges object classification, attribute identification, and pairwise visual relation tracking to compile structured, directed graph semantic charts from images.

## 📊 Computational Architecture Overview
The system processes unstructured image regions through a decoupled 3-stage neural classification framework:
1. **Object Classification Head:** Maps independent regional image embeddings directly into targeted categorical labels.
2. **Attribute Classification Head:** Uses a multi-label sigmoid layer matrix to deduce spatial surface textures and colors ($1 \rightarrow \text{Properties}$).
3. **Directed Relationship Head:** Groups object pair embedding maps side by side to extract topological, directional visual relationships ($\text{Subject} \rightarrow \text{Predicate} \rightarrow \text{Object}$).

## 🚀 Execution Instructions
```bash
# 1. Complete environment setup requirements
pip install -r requirements.txt

# 2. Trigger automated inference compilation pipeline
python src/run_inference.py