# **Manifold Learning and Graph Kernels**  
### *Enhancing Graph Classification with Dimensionality Reduction*  

---

## **📌 Table of Contents**  
1. [Project Overview](#-project-overview)  
2. [Key Features](#-key-features)  
3. [Algorithms](#-algorithms)  
4. [Datasets](#-datasets)  
5. [Installation](#-installation)  
6. [Usage](#-usage)  
7. [Results](#-results)  
8. [Contributors](#-contributors) 

---

## **🌐 Project Overview**  
This research explores **graph kernels** paired with **manifold learning** to boost graph classification accuracy. Implemented in Python, it evaluates:  

- **Graph Kernel**: Weisfeiler-Lehman (WL) subtree kernel  
- **Manifold Techniques**: Isomap & Locally Linear Embedding (LLE)  

Tested on:  
🔬 **PPI**: Protein-protein interaction networks  
🖼️ **SHOCK**: 2D shape skeletons  

---

## **✨ Key Features**  

### **Core Components**  
| **Module** | **Capabilities** |  
|------------|------------------|  
| **WL Kernel** | Multi-iteration label propagation, parallelized graph processing |  
| **Isomap** | Geodesic distance preservation, neighborhood graph embedding |  
| **LLE** | Local linearity preservation, sparse eigenvalue decomposition |  

### **Evaluation**  
- 10-fold stratified cross-validation  
- Linear SVM classifier  
- Metrics: Accuracy (min/mean/max ± σ)  

---

## **🧮 Algorithms**  

### **Weisfeiler-Lehman Kernel**  
```python
class WeisfeilerLehman:
    def run(self):
        # 1. Initialize node labels (degrees)
        # 2. For h iterations:
        #    a. Relabel nodes via neighbor aggregation  
        #    b. Compress labels  
        # 3. Compute normalized kernel matrix
```

### **Manifold Learning**  
```python
# Isomap
embedding = Isomap(n_neighbors=12, n_components=5).fit_transform(K)

# LLE 
embedding = LocallyLinearEmbedding(n_neighbors=8, method='standard').fit_transform(K)
```

---

## **📊 Datasets**  
| Dataset | Graphs | Avg Nodes | Classes | Description |  
|---------|--------|-----------|---------|-------------|  
| PPI | 86 | ~200 | 2 | Protein interaction networks |  
| SHOCK | 150 | ~33 | 5 | 2D shape skeletons |  

**Structure:**  
```matlab
% MATLAB .mat format
G = [struct('am', adj_mat1), ...]  % Adjacency matrices
labels = [0, 1, ...]               % Class labels
```

---

## **🛠 Installation**  
1. **Requirements**:  
   ```bash
   pip install numpy scipy scikit-learn matplotlib
   ```
2. **Data Setup**:  
   ```bash
   mkdir dataset
   wget -P dataset/ http://www.dsi.unive.it/~atorsell/AI/graph/{PPI,Shock}.mat
   ```

---

## **🚀 Usage**  
### **1. Compute WL Kernel**  
```python
wl = WeisfeilerLehman(graphs, h=4)
K = wl.run()  # n×n similarity matrix
```

### **2. Dimensionality Reduction**  
```python
# Optimal params from grid search
X_iso = Isomap(n_neighbors=12, n_components=5).fit_transform(K)
```

### **3. Evaluate**  
```python
cv_scores = cross_val_score(SVC(kernel='linear'), X_iso, labels, cv=10)
print(f"Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

---

## **📈 Results**  
### **PPI Dataset**  
| Method | Accuracy | Parameters |  
|--------|----------|------------|  
| WL Only | 0.82 ± 0.05 | - |  
| WL+Isomap | **0.87 ± 0.03** | k=12, d=5 |  

### **SHOCK Dataset**  
| Method | Accuracy | Parameters |  
|--------|----------|------------|  
| WL Only | 0.76 ± 0.08 | - |  
| WL+LLE | **0.83 ± 0.06** | k=18, d=4 |  

---

## **👥 Contributors**  
- **Khushbu Mahendra Patil**
- **Vafa Khalid**

---
