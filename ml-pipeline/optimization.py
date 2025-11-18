import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Lấy đường dẫn gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class WhaleOptimizationAlgorithm:
    """Whale Optimization Algorithm for feature selection and hyperparameter tuning"""
    
    def __init__(self, search_agents_no=10, max_iter=50):
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.dim = None # Sẽ được set khi gọi optimize
        self.positions = None
        self.fitness_values = None
        self.best_agent = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_positions(self, dim):
        """Khởi tạo vị trí các agent"""
        self.dim = dim
        self.positions = np.random.uniform(0, 1, (self.search_agents_no, dim)) # Dùng 0-1 thay vì -1 đến 1 để dễ mapping
        self.fitness_values = np.zeros(self.search_agents_no)
        
    def fitness_function(self, position, X, y):
        """Hàm fitness: Minimize Error Rate + Feature Penalty"""
        try:
            # Feature selection mask (Threshold 0.5)
            feature_mask = position[:X.shape[1]] > 0.5
            selected_features = np.sum(feature_mask)
            
            if selected_features == 0:
                return float('inf') # Phạt nặng nếu không chọn feature nào
            
            # Hyperparameters từ 2 chiều cuối
            # n_estimators: 50 - 200
            n_estimators = int(50 + 150 * position[X.shape[1]])
            # max_depth: 5 - 25
            max_depth = int(5 + 20 * position[X.shape[1] + 1])
            
            # Giới hạn giá trị hợp lệ
            n_estimators = max(50, min(200, n_estimators))
            max_depth = max(5, min(25, max_depth))
            
            # Chọn features
            X_selected = X.iloc[:, feature_mask] if hasattr(X, 'iloc') else X[:, feature_mask]
            
            # Train model nhanh (n_jobs=-1 để tận dụng CPU)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation score (Accuracy)
            scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy') # cv=3 cho nhanh
            accuracy = np.mean(scores)
            
            # Fitness = Error Rate + Penalty (ít feature càng tốt)
            alpha = 0.99 # Trọng số cho accuracy
            beta = 0.01  # Trọng số cho số lượng feature
            fitness = alpha * (1 - accuracy) + beta * (selected_features / X.shape[1])
            
            return fitness
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, X, y):
        """Thực hiện optimization"""
        # Dim = Số lượng features + 2 tham số (n_estimators, max_depth)
        dim = X.shape[1] + 2  
        self.initialize_positions(dim)
        
        print("🐋 Starting Whale Optimization Algorithm...")
        
        for iteration in range(self.max_iter):
            a = 2 - iteration * (2 / self.max_iter)  # a giảm từ 2 về 0
            
            for i in range(self.search_agents_no):
                # Boundary check (kẹp giá trị trong [0, 1])
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                
                # Calculate fitness
                self.fitness_values[i] = self.fitness_function(self.positions[i], X, y)
                
                # Update best agent
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_agent = self.positions[i].copy()
            
            # Update positions
            for i in range(self.search_agents_no):
                r1 = np.random.random()
                r2 = np.random.random()
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                b = 1
                l = (np.random.random() * 2) - 1
                
                p = np.random.random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = abs(C * self.best_agent - self.positions[i])
                        self.positions[i] = self.best_agent - A * D
                    else:
                        # Search for prey (Random agent)
                        rand_idx = np.random.randint(0, self.search_agents_no)
                        random_agent = self.positions[rand_idx]
                        D = abs(C * random_agent - self.positions[i])
                        self.positions[i] = random_agent - A * D
                else:
                    # Bubble-net attacking (Spiral)
                    distance_to_best = abs(self.best_agent - self.positions[i])
                    self.positions[i] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_agent
            
            self.convergence_curve.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_agent, self.best_fitness


class SquirrelSearchAlgorithm:
    """Squirrel Search Algorithm for enhanced optimization"""
    
    def __init__(self, n_squirrels=10, max_iter=50):
        self.n_squirrels = n_squirrels
        self.max_iter = max_iter
        self.dim = None
        self.positions = None
        self.fitness_values = None
        self.best_squirrel = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.woa_helper = WhaleOptimizationAlgorithm() # Mượn hàm fitness của WOA cho đồng bộ
        
    def initialize_positions(self, dim):
        self.dim = dim
        self.positions = np.random.uniform(0, 1, (self.n_squirrels, dim))
        self.fitness_values = np.zeros(self.n_squirrels)
        
    def optimize(self, X, y):
        """Thực hiện SSA optimization"""
        dim = X.shape[1] + 2
        self.initialize_positions(dim)
        
        print("🐿️ Starting Squirrel Search Algorithm...")
        
        for iteration in range(self.max_iter):
            # 1. Tính fitness
            for i in range(self.n_squirrels):
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                self.fitness_values[i] = self.woa_helper.fitness_function(self.positions[i], X, y)
                
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_squirrel = self.positions[i].copy()
            
            # 2. Sắp xếp sóc (Squirrels)
            sorted_indices = np.argsort(self.fitness_values)
            # Vị trí tốt nhất (Hickory tree)
            hickory_pos = self.positions[sorted_indices[0]]
            # 3 vị trí tiếp theo (Acorn trees)
            acorn_pos = self.positions[sorted_indices[1:4]] if self.n_squirrels >= 4 else [hickory_pos]
            
            # 3. Cập nhật vị trí (Simplified SSA Logic)
            dg = np.random.random() # Gliding distance
            gc = 1.9 # Gliding constant
            
            for i in range(self.n_squirrels):
                if i == sorted_indices[0]: 
                    # Con sóc giỏi nhất: Nhảy ngẫu nhiên nhỏ để tìm cục bộ (Exploitation)
                    self.positions[i] += np.random.normal(0, 0.01, self.dim)
                elif i < 4: 
                    # Các con sóc giỏi nhì: Di chuyển về hướng con giỏi nhất
                    if np.random.random() >= 0.5: # Pdp
                        self.positions[i] += dg * gc * (hickory_pos - self.positions[i])
                else:
                    # Các con sóc còn lại: Di chuyển về hướng Acorn trees
                    target = acorn_pos[i % len(acorn_pos)]
                    if np.random.random() >= 0.5:
                        self.positions[i] += dg * gc * (target - self.positions[i])

            self.convergence_curve.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_squirrel, self.best_fitness


class WOA_SSA_Hybrid:
    """Hybrid WOA-SSA Optimization Algorithm"""
    
    def __init__(self, population_size=20, max_iter=50):
        self.woa = WhaleOptimizationAlgorithm(search_agents_no=population_size//2, max_iter=max_iter)
        self.ssa = SquirrelSearchAlgorithm(n_squirrels=population_size//2, max_iter=max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.feature_count = 0 # Để lưu số lượng features
        
    def optimize(self, X, y):
        """Thực hiện hybrid optimization"""
        print("🔗 Starting WOA-SSA Hybrid Optimization...")
        
        # Chạy song song (hoặc tuần tự)
        woa_solution, woa_fitness = self.woa.optimize(X, y)
        ssa_solution, ssa_fitness = self.ssa.optimize(X, y)
        
        # So sánh và chọn
        if woa_fitness < ssa_fitness:
            self.best_solution = woa_solution
            self.best_fitness = woa_fitness
            print("✅ WOA performed better")
        else:
            self.best_solution = ssa_solution
            self.best_fitness = ssa_fitness
            print("✅ SSA performed better")
        
        self.feature_count = X.shape[1] # Lưu lại số lượng features
        print(f"🎯 Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness
    
    def plot_convergence(self):
        """Vẽ convergence curve"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.woa.convergence_curve, label='WOA', linewidth=2, marker='o', markevery=5)
        plt.plot(self.ssa.convergence_curve, label='SSA', linewidth=2, marker='s', markevery=5)
        
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value (Lower is Better)')
        plt.title('WOA-SSA Hybrid Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(RESULTS_DIR, 'optimization_convergence.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Convergence plot saved to: {save_path}")
        plt.show()
        plt.close()
    
    def get_optimized_model(self, X, y):
        """Train model với optimized parameters"""
        if self.best_solution is None:
            print("⚠️ Chưa chạy optimize(). Đang chạy ngay...")
            self.optimize(X, y)

        # Giải mã giải pháp tốt nhất
        feature_mask = self.best_solution[:X.shape[1]] > 0.5
        n_estimators, max_depth = self.get_optimized_hyperparameters()
        
        # Chọn feature
        X_optimized = X.iloc[:, feature_mask] if hasattr(X, 'iloc') else X[:, feature_mask]
        
        print(f"🔧 Optimized Parameters:")
        print(f"   Selected Features: {np.sum(feature_mask)}/{X.shape[1]}")
        print(f"   n_estimators: {n_estimators}")
        print(f"   max_depth: {max_depth}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_optimized, y)
        return model, feature_mask
    
    def get_optimized_hyperparameters(self):
        """Lấy optimized hyperparameters"""
        # idx_n_est = self.feature_count
        # idx_depth = self.feature_count + 1
        
        # Do self.best_solution có độ dài = feature_count + 2
        # 2 phần tử cuối cùng là hyperparameters
        val_n_est = self.best_solution[-2]
        val_depth = self.best_solution[-1]
        
        n_estimators = int(50 + 150 * val_n_est)
        max_depth = int(5 + 20 * val_depth)
        
        return n_estimators, max_depth