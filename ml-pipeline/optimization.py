import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# --- CẤU HÌNH MATPLOTLIB ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------

# Đường dẫn lưu kết quả
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _safe_get_X_selected(X, mask):
    """Trả về view của X đã chọn cột, hỗ trợ pandas DataFrame và numpy array."""
    if hasattr(X, 'iloc'):
        return X.iloc[:, mask]
    else:
        return X[:, mask]


class WhaleOptimizationAlgorithm:
    def __init__(self, search_agents_no=10, max_iter=50, cv_splits=3, random_state=42):
        self.search_agents_no = max(2, int(search_agents_no))
        self.max_iter = int(max_iter)
        self.dim = None
        self.positions = None
        self.fitness_values = None
        self.best_agent = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.random_state = random_state

    def initialize_positions(self, dim, init_positions=None):
        self.dim = dim
        if init_positions is None:
            self.positions = np.random.RandomState(self.random_state).uniform(0, 1, (self.search_agents_no, dim))
        else:
            # nếu truyền một số điểm ban đầu (shape must match)
            self.positions = np.asarray(init_positions).copy()
            if self.positions.shape[0] != self.search_agents_no:
                # nếu ít hơn, thêm random
                rng = np.random.RandomState(self.random_state)
                extra = rng.uniform(0, 1, (self.search_agents_no - self.positions.shape[0], dim))
                self.positions = np.vstack([self.positions, extra])
        self.fitness_values = np.full(self.search_agents_no, np.inf)

    def fitness_function(self, position, X, y):
        # position: vector dim = n_features + 2
        try:
            n_feat = X.shape[1]
            feature_mask = position[:n_feat] > 0.5
            selected_features = int(np.sum(feature_mask))
            if selected_features == 0:
                return float('inf')

            # 2 biến cuối cùng là hyperparams
            idx_n_est = n_feat
            idx_max_depth = n_feat + 1
            n_estimators = int(50 + 150 * float(position[idx_n_est]))
            max_depth = int(5 + 20 * float(position[idx_max_depth]))
            n_estimators = max(50, min(200, n_estimators))
            max_depth = max(5, min(25, max_depth))

            X_selected = _safe_get_X_selected(X, feature_mask)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            scores = cross_val_score(model, X_selected, y, cv=self.cv, scoring='accuracy', n_jobs=-1)

            # fitness: accuracy (to minimize -> 1-acc) + small penalty for more features
            alpha = 0.98
            beta = 0.02
            return alpha * (1.0 - float(np.mean(scores))) + beta * (selected_features / float(n_feat))
        except Exception:
            return float('inf')

    def optimize(self, X, y, init_positions=None, verbose=True):
        dim = X.shape[1] + 2
        self.initialize_positions(dim, init_positions=init_positions)
        rng = np.random.RandomState(self.random_state)
        if verbose:
            print("Starting Whale Optimization Algorithm...")

        for iteration in range(self.max_iter):
            # linearly decreasing a
            a = 2.0 - iteration * (2.0 / max(1, self.max_iter))

            # evaluate
            for i in range(self.search_agents_no):
                # clip to [0,1]
                self.positions[i] = np.clip(self.positions[i], 0.0, 1.0)
                self.fitness_values[i] = self.fitness_function(self.positions[i], X, y)
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_agent = self.positions[i].copy()

            # update positions
            for i in range(self.search_agents_no):
                r1 = rng.random(); r2 = rng.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                b = 1.0
                l = (rng.random() * 2.0) - 1.0
                p = rng.random()

                if p < 0.5:
                    if abs(A) < 1.0:
                        D = np.abs(C * self.best_agent - self.positions[i])
                        self.positions[i] = self.best_agent - A * D
                    else:
                        rand_idx = rng.randint(0, self.search_agents_no)
                        self.positions[i] = self.positions[rand_idx] - A * np.abs(C * self.positions[rand_idx] - self.positions[i])
                else:
                    # spiral
                    distance2best = np.abs(self.best_agent - self.positions[i])
                    self.positions[i] = distance2best * np.exp(b * l) * np.cos(2.0 * np.pi * l) + self.best_agent

            # ensure valid range after update
            self.positions = np.clip(self.positions, 0.0, 1.0)

            self.convergence_curve.append(self.best_fitness)
            if verbose and ((iteration + 1) % 10 == 0 or iteration == 0):
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6f}")

        return self.best_agent, self.best_fitness


class SquirrelSearchAlgorithm:
    def __init__(self, n_squirrels=10, max_iter=50, random_state=42):
        self.n_squirrels = max(2, int(n_squirrels))
        self.max_iter = int(max_iter)
        self.dim = None
        self.positions = None
        self.fitness_values = None
        self.best_squirrel = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.random_state = random_state

    def initialize_positions(self, dim, base=None, spread=0.2):
        self.dim = dim
        rng = np.random.RandomState(self.random_state)
        if base is None:
            self.positions = rng.uniform(0, 1, (self.n_squirrels, dim))
        else:
            base = np.asarray(base)
            # tạo dân cư quanh base bằng gaussian noise, clip vào [0,1]
            noise = rng.normal(loc=0.0, scale=spread, size=(self.n_squirrels, dim))
            self.positions = np.clip(base + noise, 0.0, 1.0)
        self.fitness_values = np.full(self.n_squirrels, np.inf)

    def optimize(self, X, y, fitness_func, init_base=None, verbose=True):
        dim = X.shape[1] + 2
        if init_base is None:
            self.initialize_positions(dim)
        else:
            self.initialize_positions(dim, base=init_base)

        rng = np.random.RandomState(self.random_state)
        if verbose:
            print("Starting Squirrel Search Algorithm (refinement)...")

        for iteration in range(self.max_iter):
            # evaluate
            for i in range(self.n_squirrels):
                self.positions[i] = np.clip(self.positions[i], 0.0, 1.0)
                self.fitness_values[i] = fitness_func(self.positions[i], X, y)
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_squirrel = self.positions[i].copy()

            # Movement rules (improved, allow exploration)
            sorted_idx = np.argsort(self.fitness_values)
            leader = self.positions[sorted_idx[0]]

            for k in range(self.n_squirrels):
                if k == sorted_idx[0]:
                    # leader does small local exploitation
                    self.positions[k] += 0.05 * rng.random() * (leader - self.positions[k])
                else:
                    r = rng.random()
                    if r < 0.6:
                        # move towards leader (exploitation)
                        self.positions[k] += 0.4 * rng.random() * (leader - self.positions[k])
                    elif r < 0.85:
                        # random relocation (escape local minima)
                        self.positions[k] = np.clip(self.positions[k] + rng.normal(0, 0.1, size=self.dim), 0.0, 1.0)
                    else:
                        # small random exploration
                        self.positions[k] += 0.1 * (rng.random(self.dim) - 0.5)

            # occasional predator attack simulation: worst squirrels relocate
            if iteration % max(1, int(self.max_iter / 10)) == 0:
                worst = sorted_idx[-max(1, int(0.1 * self.n_squirrels)):]
                for w in worst:
                    self.positions[w] = np.clip(self.positions[w] + rng.normal(0, 0.15, size=self.dim), 0.0, 1.0)

            self.positions = np.clip(self.positions, 0.0, 1.0)

            self.convergence_curve.append(self.best_fitness)
            if verbose and ((iteration + 1) % 10 == 0 or iteration == 0):
                print(f"   Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6f}")

        return self.best_squirrel, self.best_fitness


class WOA_SSA_Hybrid:
    """Hybrid cooperative: chạy WOA để tìm giải pháp tốt, sau đó SSA refine quanh best của WOA.
    - Thiết kế đơn giản, ổn định, phù hợp cho kích thước feature trung bình.
    """
    def __init__(self, population_size=20, max_iter=50, cv_splits=3, random_state=42):
        self.population_size = max(4, int(population_size))
        self.max_iter = int(max_iter)
        # chia cho WOA và SSA
        half = max(2, self.population_size // 2)
        self.woa = WhaleOptimizationAlgorithm(search_agents_no=half, max_iter=max_iter, cv_splits=cv_splits, random_state=random_state)
        self.ssa = SquirrelSearchAlgorithm(n_squirrels=half, max_iter=max_iter, random_state=random_state + 1)
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self, X, y, verbose=True):
        # 1) Chạy WOA để khám phá
        woa_best, woa_fit = self.woa.optimize(X, y, verbose=verbose)

        # 2) Khởi tạo SSA quanh giải pháp tốt nhất của WOA để refine (local search)
        ssa_best, ssa_fit = self.ssa.optimize(X, y, fitness_func=self.woa.fitness_function, init_base=woa_best, verbose=verbose)

        # Chọn kết quả tốt hơn
        if woa_fit <= ssa_fit:
            self.best_solution = woa_best
            self.best_fitness = woa_fit
            if verbose:
                print("WOA result selected as final best.")
        else:
            self.best_solution = ssa_best
            self.best_fitness = ssa_fit
            if verbose:
                print("SSA refinement improved WOA; SSA result selected as final best.")

        # merge convergence curves for plotting convenience
        # pad shorter curve with its last value
        len_w = len(self.woa.convergence_curve)
        len_s = len(self.ssa.convergence_curve)
        maxlen = max(len_w, len_s)
        wcurve = np.array(self.woa.convergence_curve + [self.woa.convergence_curve[-1]] * (maxlen - len_w))
        scurve = np.array(self.ssa.convergence_curve + [self.ssa.convergence_curve[-1]] * (maxlen - len_s))
        self.woa.convergence_curve = wcurve.tolist()
        self.ssa.convergence_curve = scurve.tolist()

        if verbose:
            print(f"Best Fitness (hybrid): {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness

    def plot_convergence(self, filename='optimization_convergence.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.woa.convergence_curve, label='WOA', linewidth=2, marker='o', markevery=5)
        plt.plot(self.ssa.convergence_curve, label='SSA', linewidth=2, marker='s', markevery=5)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title('WOA-SSA Hybrid Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        outpath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(outpath, dpi=300)
        plt.close()
        return outpath

    def get_optimized_model(self, X, y):
        if self.best_solution is None:
            self.optimize(X, y, verbose=False)

        n_feat = X.shape[1]
        feature_mask = self.best_solution[:n_feat] > 0.5
        n_est = int(50 + 150 * float(self.best_solution[n_feat]))
        n_est = max(50, min(200, n_est))
        depth = int(5 + 20 * float(self.best_solution[n_feat + 1]))
        depth = max(5, min(25, depth))

        X_opt = _safe_get_X_selected(X, feature_mask)
        print(f"Optimized Parameters: Features={int(np.sum(feature_mask))}/{n_feat}, n_est={n_est}, depth={depth}")

        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
        model.fit(X_opt, y)
        return model, feature_mask


# -------------------- Example usage --------------------
if __name__ == '__main__':
    # Ví dụ: tạo dataset giả để chạy thử
    from sklearn.datasets import make_classification

    X_np, y = make_classification(n_samples=300, n_features=20, n_informative=6, random_state=0)
    X = pd.DataFrame(X_np, columns=[f'f{i}' for i in range(X_np.shape[1])])

    hybrid = WOA_SSA_Hybrid(population_size=20, max_iter=40, cv_splits=3, random_state=0)
    best_sol, best_fit = hybrid.optimize(X, y, verbose=True)
    plot_path = hybrid.plot_convergence()
    print('Convergence plot saved to:', plot_path)

    model, mask = hybrid.get_optimized_model(X, y)
    print('Selected features mask sum:', int(np.sum(mask)))
    # Lưu mask nếu cần
    np.save(os.path.join(RESULTS_DIR, 'selected_features_mask.npy'), mask)
    print('Done.')
