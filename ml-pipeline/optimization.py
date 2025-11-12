import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class WhaleOptimizationAlgorithm:
    """Whale Optimization Algorithm for feature selection and hyperparameter tuning"""
    
    def __init__(self, search_agents_no=10, max_iter=50, dim=None):
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.dim = dim
        self.positions = None
        self.fitness_values = None
        self.best_agent = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_positions(self, dim):
        """Khởi tạo vị trí các agent"""
        self.dim = dim
        self.positions = np.random.uniform(-1, 1, (self.search_agents_no, dim))
        self.fitness_values = np.zeros(self.search_agents_no)
        
    def fitness_function(self, position, X, y):
        """Hàm fitness cho feature selection và hyperparameter optimization"""
        try:
            # Feature selection mask
            feature_mask = position[:X.shape[1]] > 0
            selected_features = np.sum(feature_mask)
            
            if selected_features == 0:
                return float('inf')
            
            # Hyperparameters từ position
            n_estimators = int(50 + 150 * (position[X.shape[1]] + 1) / 2)  # 50-200
            max_depth = int(5 + 20 * (position[X.shape[1] + 1] + 1) / 2)   # 5-25
            
            # Chọn features
            X_selected = X.iloc[:, feature_mask] if hasattr(X, 'iloc') else X[:, feature_mask]
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation score
            scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
            fitness = 1 - np.mean(scores)  # Minimize error
            
            # Penalty for too many features
            feature_penalty = 0.001 * selected_features / X.shape[1]
            fitness += feature_penalty
            
            return fitness
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, X, y):
        """Thực hiện optimization"""
        dim = X.shape[1] + 2  # Features + 2 hyperparameters
        self.initialize_positions(dim)
        
        print("🐋 Starting Whale Optimization Algorithm...")
        
        for iteration in range(self.max_iter):
            a = 2 - iteration * (2 / self.max_iter)  # a decreases linearly from 2 to 0
            a2 = -1 + iteration * (-1 / self.max_iter)  # a2 decreases linearly from -1 to -2
            
            for i in range(self.search_agents_no):
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
                
                p = np.random.random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey
                        D = abs(C * self.best_agent - self.positions[i])
                        self.positions[i] = self.best_agent - A * D
                    else:
                        # Search for prey
                        random_agent = self.positions[np.random.randint(0, self.search_agents_no)]
                        D = abs(C * random_agent - self.positions[i])
                        self.positions[i] = random_agent - A * D
                else:
                    # Bubble-net attacking
                    distance_to_best = abs(self.best_agent - self.positions[i])
                    self.positions[i] = distance_to_best * np.exp(0.5 * a2) * np.cos(2 * np.pi * a2) + self.best_agent
                
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], -1, 1)
            
            self.convergence_curve.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_agent, self.best_fitness
    
    def get_optimized_features(self, X):
        """Lấy feature mask từ best agent"""
        feature_mask = self.best_agent[:X.shape[1]] > 0
        return feature_mask
    
    def get_optimized_hyperparameters(self):
        """Lấy hyperparameters từ best agent"""
        n_estimators = int(50 + 150 * (self.best_agent[self.dim - 2] + 1) / 2)
        max_depth = int(5 + 20 * (self.best_agent[self.dim - 1] + 1) / 2)
        return n_estimators, max_depth

class SquirrelSearchAlgorithm:
    """Squirrel Search Algorithm for enhanced optimization"""
    
    def __init__(self, n_squirrels=10, max_iter=50, dim=None):
        self.n_squirrels = n_squirrels
        self.max_iter = max_iter
        self.dim = dim
        self.positions = None
        self.fitness_values = None
        self.best_squirrel = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_positions(self, dim):
        """Khởi tạo vị trí squirrels"""
        self.dim = dim
        self.positions = np.random.uniform(-1, 1, (self.n_squirrels, dim))
        self.fitness_values = np.zeros(self.n_squirrels)
        
    def fitness_function(self, position, X, y):
        """Hàm fitness tương tự WOA"""
        try:
            feature_mask = position[:X.shape[1]] > 0
            selected_features = np.sum(feature_mask)
            
            if selected_features == 0:
                return float('inf')
            
            n_estimators = int(50 + 150 * (position[X.shape[1]] + 1) / 2)
            max_depth = int(5 + 20 * (position[X.shape[1] + 1] + 1) / 2)
            
            X_selected = X.iloc[:, feature_mask] if hasattr(X, 'iloc') else X[:, feature_mask]
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
            fitness = 1 - np.mean(scores)
            feature_penalty = 0.001 * selected_features / X.shape[1]
            fitness += feature_penalty
            
            return fitness
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, X, y):
        """Thực hiện SSA optimization"""
        dim = X.shape[1] + 2
        self.initialize_positions(dim)
        
        print("🐿️ Starting Squirrel Search Algorithm...")
        
        for iteration in range(self.max_iter):
            # Calculate fitness
            for i in range(self.n_squirrels):
                self.fitness_values[i] = self.fitness_function(self.positions[i], X, y)
                
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_squirrel = self.positions[i].copy()
            
            # Sort squirrels by fitness
            sorted_indices = np.argsort(self.fitness_values)
            
            # Update positions (simplified SSA)
            for i in range(self.n_squirrels):
                if i == sorted_indices[0]:  # Best squirrel
                    # Explore new areas
                    self.positions[i] += np.random.normal(0, 0.1, self.dim)
                else:
                    # Move towards better squirrels
                    target_idx = sorted_indices[np.random.randint(0, 3)]  # Top 3 squirrels
                    self.positions[i] += 0.5 * (self.positions[target_idx] - self.positions[i])
                
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], -1, 1)
            
            self.convergence_curve.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_squirrel, self.best_fitness

class WOA_SSA_Hybrid:
    """Hybrid WOA-SSA Optimization Algorithm"""
    
    def __init__(self, population_size=20, max_iter=100):
        self.woa = WhaleOptimizationAlgorithm(search_agents_no=population_size//2, max_iter=max_iter)
        self.ssa = SquirrelSearchAlgorithm(n_squirrels=population_size//2, max_iter=max_iter)
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def optimize(self, X, y):
        """Thực hiện hybrid optimization"""
        print("🔗 Starting WOA-SSA Hybrid Optimization...")
        
        # Run both algorithms
        woa_solution, woa_fitness = self.woa.optimize(X, y)
        ssa_solution, ssa_fitness = self.ssa.optimize(X, y)
        
        # Select best solution
        if woa_fitness < ssa_fitness:
            self.best_solution = woa_solution
            self.best_fitness = woa_fitness
            print("✅ WOA performed better")
        else:
            self.best_solution = ssa_solution
            self.best_fitness = ssa_fitness
            print("✅ SSA performed better")
        
        print(f"🎯 Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness
    
    def plot_convergence(self):
        """Vẽ convergence curve"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.woa.convergence_curve, label='WOA', linewidth=2)
        plt.plot(self.ssa.convergence_curve, label='SSA', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title('WOA-SSA Hybrid Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('../results/optimization_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_optimized_model(self, X, y):
        """Train model với optimized parameters"""
        feature_mask = self.best_solution[:X.shape[1]] > 0
        n_estimators, max_depth = self.get_optimized_hyperparameters()
        
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
        n_estimators = int(50 + 150 * (self.best_solution[self.woa.dim - 2] + 1) / 2)
        max_depth = int(5 + 20 * (self.best_solution[self.woa.dim - 1] + 1) / 2)
        return n_estimators, max_depth