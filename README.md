# Comprehensive-Optimization-Algorithms-Theory-Implementation-Analysis

This repository presents a **full comparative study** and **implementation** of a variety of optimization algorithms ranging from **linear programming (Revised Simplex Method)**, **regularized regression techniques (Ridge, LASSO, Elastic Net)**, to **classical unconstrained nonlinear optimization methods** in both **1D and multidimensional spaces**.

All algorithms are implemented in **Python** with theoretical backgrounds, pseudocode, and performance analysis on benchmark datasets/functions. This work is based on **Math 303 - Linear and Non-linear Programming for Computational Sciences**.

---

## 📂 Repository Structure

```
Comprehensive-Optimization-Algorithms-Theory-Implementation-Analysis/
│
├── revised_simplex_lpp/
│   ├── revised_simplex_lpp_theory.pdf
│   ├── revised_simplex_lpp_code.ipynb
│
├── regularization_regression/
│   ├── ridge_lasso_elasticnet_theory.pdf
│   ├── ridge_lasso_elasticnet_code.ipynb
│
├── classical_optimization_methods/
│   ├── classical_optimization_theory.pdf
│   ├── classical_optimization_code.ipynb
│
└── README.md
```

---

## 1️⃣ Revised Simplex Method for Linear Programming Problems

### **Overview**
The **Revised Simplex Method** is an efficient variant of the Simplex algorithm that:
- Works on large datasets by updating only essential parts of the tableau.
- Focuses on the **basis matrix** \( B \) and its inverse \( B^{-1} \).
- Reduces memory and computational cost compared to the classical Simplex.

---

### **Mathematical Formulation**
**Reduced Cost Calculation:**
\[
c'_N = c_N - c_B \cdot B^{-1} N
\]
Where:
- \( N \) = matrix of non-basic variable columns  
- \( c_N \) = cost coefficients of non-basic variables  
- \( c_B \) = cost coefficients of basic variables  

**Optimality Condition (Maximization)**:
\[
\text{Optimal if} \quad c'_N \geq 0
\]

**Minimum Ratio Test**:
\[
\theta = \min\left(\frac{x_B}{D}\right)
\]
- \( x_B \) = current basic variable values  
- \( D \) = direction vector for entering variable  

---

### **Algorithm**
```
1. Start with a basic feasible solution (BFS) in standard form.
2. Compute B⁻¹.
3. Compute reduced costs for non-basic variables: Zj - Cj = CB·B⁻¹·Pj - Cj
4. If all reduced costs satisfy optimality → STOP.
5. Select entering variable (most negative reduced cost for max problems).
6. Compute direction vector: d = B⁻¹·Pj
7. Apply minimum ratio test → select leaving variable.
8. Update basis B and compute new B⁻¹.
9. Repeat until optimal solution is found.
```

---

### **Special Cases Covered**
- Regular Simplex
- Two-Phase Simplex
- Degeneracy
- Alternative Optima
- Unbounded Solution
- Infeasible Solution

---

### **References**
- Taha, H. A., *Operations Research: An Introduction*, 10th ed.  
- Boyd, S., & Vandenberghe, L., *Convex Optimization*  
- Rao, S. S., *Engineering Optimization: Theory and Practice*, 5th ed.

---

## 2️⃣ Regularization in Regression: Ridge, LASSO, Elastic Net

### **Overview**
Regularization improves model generalization by adding a penalty to large coefficients.  
This project implements and compares:
- **Ridge Regression** (L2 penalty)
- **LASSO Regression** (L1 penalty)
- **Elastic Net Regression** (Hybrid L1 + L2)

Two datasets used:
- Housing dataset
- Advertising dataset

---

### **Mathematical Models**

**Ridge**:
\[
F(w) = \text{MSE}(w) + \lambda ||w||^2
\]

**LASSO**:
\[
F(w) = \text{MSE}(w) + \lambda ||w||_1
\]

**Elastic Net**:
\[
F(w) = \text{MSE}(w) + \lambda \left( \alpha ||w||_1 + \frac{1 - \alpha}{2} ||w||^2 \right)
\]

---

### **From-Scratch Implementation Steps**
**Ridge Pseudocode:**
```
1. Add bias term to X if needed.
2. Compute regularized matrix: (XᵀX + λI)
3. Compute weights: w = (XᵀX + λI)⁻¹Xᵀy
4. Predict: y_pred = X·w
```

**LASSO Pseudocode (Coordinate Descent):**
```
Initialize β = 0
Repeat until convergence:
    For each coefficient βj:
        Compute partial residual rj = y − Xβ + Xjβj
        sj = Xjᵀrj
        βj = sign(sj)·max(|sj| − λ, 0) / (XjᵀXj)
```

**Elastic Net:**
- Similar to LASSO but with combined L1 and L2 penalties.
- Parameters λ and α tuned via cross-validation.

---

### **Comparison Table**

| Method     | Feature Selection | Handles Multicollinearity | Extra Parameters | Sparsity |
|------------|-------------------|---------------------------|------------------|----------|
| Ridge      | No                | Yes                       | λ                | No       |
| LASSO      | Yes               | Partially                 | λ                | Yes      |
| ElasticNet | Yes               | Yes                       | λ, α             | Yes      |

---

## 3️⃣ Classical Optimization Methods (1D & Multidimensional)

### **Part 1 — 1D Minimization Methods**
- **Fibonacci Method**
- **Golden Section Method**
- **Newton’s Method**
- **Quasi-Newton Method**
- **Secant Method**

---

### **Part 2 — Multidimensional Unconstrained Optimization**
- **Fletcher–Reeves Conjugate Gradient**
- **Marquardt Method**
- **Quasi-Newton (BFGS)**

---

### **Benchmark Functions**
**Rosenbrock Function:**
\[
f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2
\]
Start: \( X_0 = (-1.2, 1.0) \)

**Powell’s Quartic Function:**
\[
f(x_1, x_2, x_3, x_4) = (x_1 + 10x_2)^2 + 5(x_3 - x_4)^2 + (x_2 - 2x_3)^4 + 10(x_1 - x_4)^4
\]
Start: \( X_0 = (3.0, -1.0, 0.0, 1.0) \)

---

### **Results — Rosenbrock**
| Method           | Iterations | Optimal Solution         | Optimal Value      | CPU Time (s) |
|------------------|-----------:|-------------------------|--------------------|--------------|
| Fletcher-Reeves  | 41         | [1.00000291, 1.00000585] | 8.563×10⁻¹²         | 0.018179     |
| Marquardt        | 41         | [0.99999998, 0.99999996] | 4.866×10⁻¹⁶         | 0.003407     |
| BFGS             | 18         | [1.0, 1.0]               | 1.010×10⁻¹⁸         | 0.007212     |

---

### **Results — Powell**
| Method           | Iterations | Optimal Value            | CPU Time (s) |
|------------------|-----------:|--------------------------|--------------|
| Fletcher-Reeves  | 1000       | 9.061×10⁻¹⁰               | 0.341768     |
| Marquardt        | 100        | 2.093×10                  | 0.004395     |
| BFGS             | 32         | 1.167×10⁻¹⁰               | 0.008371     |

---

### **Strengths & Weaknesses Summary**
| Method        | Strengths                                | Weaknesses                         |
|---------------|------------------------------------------|-------------------------------------|
| Fibonacci     | Guaranteed convergence, derivative-free  | Needs Fibonacci precomputation      |
| GoldenSection | Simple, derivative-free                  | Slightly slower than Fibonacci      |
| Newton        | Quadratic convergence                    | Requires Hessian                    |
| Quasi-Newton  | No Hessian, superlinear convergence      | More costly than Newton             |
| Secant        | Derivative-free                          | Can diverge with poor guess         |
| CG (FR)       | Large-scale efficiency                   | Sensitive to line search accuracy   |
| Marquardt     | Robust, blends Newton & descent          | λ tuning needed                     |
| BFGS          | Fast, robust                             | Matrix ops costly in high dimensions|

---

## ⚙️ How to Run

### **Requirements**
- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib
```

### **Running the Notebooks**
```bash
jupyter notebook
```
Open the `.ipynb` files in the corresponding folders.

---

## 📚 References
- S. Boyd, L. Vandenberghe, *Convex Optimization*
- S. S. Rao, *Engineering Optimization: Theory and Practice*
- H. A. Taha, *Operations Research: An Introduction*
- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*
