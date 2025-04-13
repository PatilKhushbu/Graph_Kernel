# **Manifold Learning and Graph Kernels**  
### *Enhancing Graph Classification with Dimensionality Reduction*  

---

## **📌 Table of Contents**  
1. [Project Overview](#-project-overview)  
2. [Key Features](#-key-features)  
3. [Algorithms Implemented](#-algorithms-implemented)  
4. [Datasets](#-datasets)  
5. [Installation](#-installation)  
6. [Usage](#-usage)  
7. [Results](#-results)  
8. [Contributors](#-contributors)
   
---

## **🌐 Project Overview**  
This project explores the combination of **graph kernels** and **manifold learning** techniques to improve graph classification performance. Implemented in Python, it compares:  

- **Graph Kernels**: Weisfeiler-Lehman Kernel  
- **Manifold Learning**: Isomap & Locally Linear Embedding (LLE)  

Tested on two benchmark datasets:  
1. Protein-Protein Interaction (PPI) networks  
2. 2D shape skeletons (SHOCK)  

Based on research from **Ca' Foscari University of Venice** (Information Retrieval course CM0473).  

---

## **✨ Key Features**  

### **Graph Kernel Implementation**  
| Component | Specification |  
|-----------|---------------|  
| Algorithm | Weisfeiler-Lehman (WL) subtree kernel |  
| Iterations | Configurable (default h=4) |  
| Parallelism | Multi-threaded graph processing |  
| Output | Normalized similarity matrices |  

### **Manifold Learning**  
| Technique | Parameters |  
|-----------|------------|  
| Isomap | Neighborhood size (2-24), Components (2-9) |  
| LLE | Neighborhood size (2-24), Components (2-9) |  

### **Evaluation Framework**  
- 10-fold stratified cross-validation  
- Linear SVM classifier  
- Comprehensive metrics: Min/Mean/Max accuracy ± Std. Dev.  

---

## **🧮 Algorithms Implemented**  

### **Weisfeiler-Lehman Kernel**  
```python
class WeisfeilerLehman:
    def __init__(self, graphs, h):
        # Initializes node labels (degrees) and compression dictionary
        ...
        
    def run(self):
        # Executes h WL iterations and computes kernel matrix
        ...
```

### **Manifold Learning**  
```python
# Isomap Transformation
manifold.Isomap(n_neighbors=5, n_components=3).fit_transform(matrix)

# LLE Transformation  
manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=4).fit_transform(matrix)
```

---

## **📊 Datasets**  

| Dataset | Type | Nodes (avg) | Graphs | Classes |  
|---------|------|------------|--------|---------|  
| [PPI](http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat) | Protein Networks | ~200 | 86 | 2 |  
| [SHOCK](http://www.dsi.unive.it/~atorsell/AI/graph/Shock.mat) | 2D Shapes | ~33 | 150 | 5 |  

**Data Structure:**  
```matlab
G = [ 
    struct('am', adjacency_matrix1, 'nl', node_labels1), 
    struct('am', adjacency_matrix2, 'nl', node_labels2),
    ...
]
labels = [class1, class2, ...] 
```

---

## **🛠 Installation**  

### **Requirements**  
- Python 3.8+  
- Libraries:  
  ```bash
  pip install numpy scipy scikit-learn matplotlib threadpoolctl
  ```

### **Data Preparation**  
1. Download datasets:  
   ```bash
   mkdir dataset && cd dataset
   wget http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat
   wget http://www.dsi.unive.it/~atorsell/AI/graph/Shock.mat
   ```

---

## **🚀 Usage**  

### **1. Compute WL Kernel Matrices**  
```python
wl = WeisfeilerLehman(graphs, h=4)
similarity_matrix = wl.run()  # Returns normalized n×n matrix
```

### **2. Apply Manifold Learning**  
```python
# Optimal parameters from grid search
isomap_embedding = Isomap(n_neighbors=8, n_components=5).fit_transform(similarity_matrix)
```

### **3. Evaluate Classifier**  
```python
scores = cross_val_score(SVC(kernel='linear'), embeddings, labels, cv=10)
print(f"Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
```

---

## **📈 Results**  

### **PPI Dataset Performance**  
| Method | Best Accuracy | Parameters |  
|--------|--------------|------------|  
| WL Only | 0.82 ± 0.05 | - |  
| WL + Isomap | 0.87 ± 0.03 | neighbors=12, components=7 |  
| WL + LLE | 0.84 ± 0.04 | neighbors=5, components=3 |  

### **SHOCK Dataset Performance**  
| Method | Best Accuracy | Parameters |  
|--------|--------------|------------|  
| WL Only | 0.76 ± 0.08 | - |  
| WL + Isomap | 0.83 ± 0.06 | neighbors=18, components=4 |  
| WL + LLE | 0.79 ± 0.07 | neighbors=9, components=5 |  

---

## **👥 Contributors**  
**Ca' Foscari University of Venice**  
- **Khushbu Mahendra Patil**
- **Vafa Khalid**
