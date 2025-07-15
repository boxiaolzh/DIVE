import numpy as np
import torch
import time
import warnings
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from src.DIVE.MI_calc import calculate_scores


class DynamicDimBO:
    def __init__(self, total_dim, objective_function, initial_samples, bounds, seed, constraints=None):
        self.seed = seed
        self.total_dim = total_dim
        self.objective_function = objective_function
        self.bounds = bounds
        self.constraints = constraints

        # Data storage
        self.X = []
        self.y = []
        self.raw_y = []
        self.simulation_count = 0
        self.X_active = []

        self.initial_samples = initial_samples
        self.curr_iter = 0

        self.all_dims = list(range(total_dim))
        self.dim_scores = np.zeros(total_dim)

        # Unified cache markers
        self._model_outdated = True
        self._data_outdated = True

        # Adaptive initialization
        self._adaptive_initialize(initial_samples=initial_samples)

        self.current_active_dim = 0
        self.active_dims = []
        self.current_active_dim = self._calculate_delta_d()

        self.noise = 5e-6
        self.sigma = 2.0
        # Only keep M matrix, since S in paper is actually M
        self.M = [np.eye(self.current_active_dim)]

        self.best_feasible_current = float(1)
        self.best_feasible_point = None

        self._select_initial_dimensions()
        self._initialize_kernel_with_mi()

        self.x_mean = None
        self.x_std = None
        self._update_normalization_params()

        # Variables for incremental update support
        self._last_active_dims = []

    def _initialize_kernel_with_mi(self):
        """Initialize kernel matrix based on MI scores"""
        if hasattr(self, 'dim_scores') and len(self.active_dims) > 0:
            active_scores = self.dim_scores[self.active_dims].copy()
            active_scores = np.clip(active_scores, 1e-3, None)
            mean_score = np.mean(active_scores)
            normalized_mi = active_scores / mean_score

            new_dim = self.current_active_dim
            new_M = np.eye(new_dim)

            # Diagonal elements: Mᵢᵢ = Ŝ(xᵢ)
            for i in range(new_dim):
                new_M[i, i] = normalized_mi[i]

            # Off-diagonal elements: Paper formula 18
            if len(self.X) > 10:
                # When sufficient data: Mᵢⱼ = √(MᵢᵢMⱼⱼ) · (1 - ρᵢⱼ · σᵢⱼ)
                X_array = np.array(self.X)
                for i in range(new_dim):
                    for j in range(i + 1, new_dim):
                        try:
                            # Calculate correlation coefficient ρᵢⱼ
                            rho_ij = abs(np.corrcoef(X_array[:, self.active_dims[i]],
                                                     X_array[:, self.active_dims[j]])[0, 1])
                        except:
                            rho_ij = 0.0

                        # Calculate interaction factor σᵢⱼ
                        sigma_ij = 1.0 - abs(normalized_mi[i] - normalized_mi[j]) / max(normalized_mi[i],
                                                                                        normalized_mi[j])

                        # Paper formula 18: Mᵢⱼ = √(MᵢᵢMⱼⱼ) · (1 - ρᵢⱼ · σᵢⱼ)
                        interaction_factor = 1 - rho_ij * sigma_ij
                        M_ij = interaction_factor * np.sqrt(new_M[i, i] * new_M[j, j])
                        new_M[i, j] = new_M[j, i] = M_ij
            else:
                # Insufficient data: set off-diagonal elements to 0 (independence assumption)
                print(
                    f"Insufficient data ({len(self.X)} <= 10), off-diagonal elements set to 0 (independence assumption)")
                for i in range(new_dim):
                    for j in range(new_dim):
                        if i != j:
                            new_M[i, j] = 0.0

            # Numerical stability
            new_M = (new_M + new_M.T) / 2
            new_M += 1e-6 * np.eye(new_dim)

            # Ensure positive definiteness
            eigenvals = np.linalg.eigvals(new_M)
            if np.min(eigenvals) <= 1e-6:
                print(f"Warning: M matrix near singular, min eigenvalue: {np.min(eigenvals):.2e}")
                new_M += (1e-6 - np.min(eigenvals)) * np.eye(new_dim)

            self.M[0] = new_M
            print(f"Kernel matrix initialization complete, M matrix dimension: {new_dim}")

    def _update_normalization_params(self):
        if not self.X:
            self.x_mean = np.zeros(self.current_active_dim)
            self.x_std = np.ones(self.current_active_dim)
            return

        X_active = self._update_active_data()
        X_array = np.array(X_active)

        self.x_mean = np.median(X_array, axis=0)
        q75, q25 = np.percentile(X_array, [75, 25], axis=0)
        self.x_std = np.maximum(q75 - q25, 1e-6)

    def _update_active_data(self):
        self.X_active = []
        for x in self.X:
            x_active = np.array([x[i] for i in self.active_dims])
            self.X_active.append(x_active)
        return self.X_active

    def _calculate_marginal_information_gain(self, candidate_dims):
        """Calculate marginal information gain for candidate dimensions"""
        if not self.active_dims:
            return {dim: self.dim_scores[dim] for dim in candidate_dims}

        marginal_gains = {}
        X_array = np.array(self.X)

        for dim in candidate_dims:
            base_info = self.dim_scores[dim]
            redundancy_scores = []

            for active_dim in self.active_dims:
                corr = abs(np.corrcoef(X_array[:, dim], X_array[:, active_dim])[0, 1])
                redundancy_scores.append(corr)

            redundancy = max(redundancy_scores) if redundancy_scores else 0.0
            marginal_gain = base_info * (1 - redundancy)
            marginal_gains[dim] = marginal_gain

        return marginal_gains

    def _calculate_information_ratio(self):
        """Calculate information ratio Ir = max(ΔI(xi)) / S̄(A)"""
        inactive_dims = [d for d in range(self.total_dim) if d not in self.active_dims]
        if not inactive_dims:
            return 0.0, {}

        marginal_gains = self._calculate_marginal_information_gain(inactive_dims)
        if not marginal_gains:
            return 0.0, {}

        max_marginal_gain = max(marginal_gains.values())
        active_avg_info = np.mean([self.dim_scores[d] for d in self.active_dims]) if self.active_dims else 1.0

        Ir = max_marginal_gain / active_avg_info if active_avg_info > 0 else 0.0
        return Ir, marginal_gains

    def _calculate_delta_d(self):
        """Paper formula: Δd* = ⌈ln(D/d) × Ir⌉"""
        Ir, marginal_gains = self._calculate_information_ratio()

        D = self.total_dim
        d = max(self.current_active_dim, 1)

        ln_ratio = np.log(D / d)
        delta_d_raw = ln_ratio * Ir
        delta_d = int(np.ceil(delta_d_raw))

        D_remaining = self.total_dim - self.current_active_dim
        delta_d = max(1, min(delta_d, D_remaining))

        print(f"  D={D}, d={d}, ln(D/d)={ln_ratio:.3f}, Ir={Ir:.3f}, Δd*={delta_d}")
        return delta_d

    def _update_mi_scores(self):
        """Update MI scores using calculate_scores function"""
        if not self.X or not self.raw_y:
            return

        try:
            # Use all available samples
            X = np.array(self.X)

            processed_outputs = []
            for y_raw in self.raw_y:
                if isinstance(y_raw, list):
                    y_raw = y_raw[0]
                if isinstance(y_raw, torch.Tensor):
                    y_raw = y_raw.numpy()
                if len(y_raw.shape) == 1:
                    y_raw = y_raw.reshape(1, -1)
                processed_outputs.append(y_raw)

            y_array = np.array(processed_outputs).reshape(len(processed_outputs), -1)

            # Count constraint satisfaction
            gain_num = phase_num = gbw_num = 0
            for y in processed_outputs:
                if y[0, 0] >= self.constraints.get('gain', 0):
                    gain_num += 1
                if y[0, 2] >= self.constraints.get('phase', 0):
                    phase_num += 1
                if y[0, 3] >= self.constraints.get('gbw', 0):
                    gbw_num += 1

            scores = calculate_scores(
                dbx=X, dby=y_array, gain_num=gain_num, GBW_num=gbw_num,
                phase_num=phase_num, iter=getattr(self, 'curr_iter', 0),
                init_num=self.initial_samples, input_dim=self.total_dim
            )

            if isinstance(scores, torch.Tensor):
                scores = scores.numpy()

            self.dim_scores = np.maximum(scores, 1e-6)

        except Exception as e:
            print(f"MI score update failed: {e}")
            if not hasattr(self, 'dim_scores'):
                self.dim_scores = np.ones(self.total_dim) / self.total_dim

    def _select_initial_dimensions(self):
        self.current_active_dim = min(self.current_active_dim, self.total_dim)
        sorted_dims = np.argsort(-self.dim_scores)
        self.active_dims = sorted_dims[:self.current_active_dim].tolist()

    def _reorder_dimensions(self):
        """Reorder dimensions based on MI scores"""
        start_time = time.time()
        max_time = 20

        try:
            sorted_dims = np.argsort(-self.dim_scores)
            new_active_dims = sorted_dims[:self.current_active_dim].tolist()

            if time.time() - start_time > max_time:
                print("Dimension reordering timeout, maintaining original dimensions")
                return

            if set(new_active_dims) == set(self.active_dims):
                return

            change_count = len(set(new_active_dims) - set(self.active_dims))
            if change_count > self.current_active_dim // 2 and self.current_active_dim > 4:
                print(f"Large dimension change detected, using progressive update")
                to_keep = self.current_active_dim - min(self.current_active_dim // 2, 3)
                importance_order = sorted(range(len(self.active_dims)),
                                          key=lambda i: self.dim_scores[self.active_dims[i]], reverse=True)
                keep_dims = [self.active_dims[i] for i in importance_order[:to_keep]]
                remaining_dims = [d for d in new_active_dims if d not in keep_dims]
                remaining_slots = self.current_active_dim - len(keep_dims)
                new_active_dims = keep_dims + remaining_dims[:remaining_slots]

            print(f"Reordering dimensions from {self.active_dims} to {new_active_dims}")
            old_active_dims = self.active_dims.copy()
            self.active_dims = new_active_dims

            if time.time() - start_time > max_time:
                print("Dimension reordering timeout, restoring original dimensions")
                self.active_dims = old_active_dims
                return

            try:
                self._reinitialize_kernel_matrices()
                print("Reordering: kernel matrix re-initialization complete")
            except Exception as e:
                print(f"Reordering kernel matrix initialization failed: {e}")
                self.active_dims = old_active_dims

        except Exception as e:
            print(f"Dimension reordering exception: {e}")

    def _reinitialize_kernel_matrices(self):
        """Re-initialize kernel matrices according to paper formulas 16-18"""
        new_dim = len(self.active_dims)
        if new_dim == 0:
            return

        # Build metric matrix M (covariance scale matrix S in paper)
        new_M = np.eye(new_dim)

        if hasattr(self, 'dim_scores'):
            active_scores = self.dim_scores[self.active_dims].copy()
            active_scores = np.clip(active_scores, 1e-3, None)
            mean_score = np.mean(active_scores)
            normalized_mi = active_scores / mean_score

            # Paper formula 17: diagonal elements Mᵢᵢ = Ŝ(xᵢ)
            for i in range(new_dim):
                new_M[i, i] = normalized_mi[i]

            # Paper formula 18: off-diagonal elements Mᵢⱼ = √(MᵢᵢMⱼⱼ) · (1 - ρᵢⱼ · σᵢⱼ)
            if len(self.X) > 10:
                X_array = np.array(self.X)
                for i in range(new_dim):
                    for j in range(i + 1, new_dim):
                        try:
                            rho_ij = abs(np.corrcoef(X_array[:, self.active_dims[i]],
                                                     X_array[:, self.active_dims[j]])[0, 1])
                        except:
                            rho_ij = 0.0

                        # Formula 19: interaction factor
                        sigma_ij = 1.0 - abs(normalized_mi[i] - normalized_mi[j]) / max(normalized_mi[i],
                                                                                        normalized_mi[j])

                        # Core formula: Mᵢⱼ = √(MᵢᵢMⱼⱼ) · (1 - ρᵢⱼ · σᵢⱼ)
                        interaction_factor = 1 - rho_ij * sigma_ij
                        M_ij = interaction_factor * np.sqrt(new_M[i, i] * new_M[j, j])
                        new_M[i, j] = new_M[j, i] = M_ij
            else:
                # Insufficient data: explicitly set off-diagonal elements to 0
                print(f"Insufficient data ({len(self.X)} <= 10), off-diagonal elements set to 0 (diagonal ARD mode)")
                for i in range(new_dim):
                    for j in range(new_dim):
                        if i != j:
                            new_M[i, j] = 0.0

        # Numerical stability
        new_M = (new_M + new_M.T) / 2
        new_M += 1e-6 * np.eye(new_dim)

        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(new_M)
        if np.min(eigenvals) <= 1e-6:
            print(f"Warning: M matrix near singular, min eigenvalue: {np.min(eigenvals):.2e}")
            new_M += (1e-6 - np.min(eigenvals)) * np.eye(new_dim)

        self.M[-1] = new_M

        print(f"Metric matrix M re-initialization complete, dimension: {new_dim}")

    def _incremental_update_M_matrix(self):
        """Incremental update of metric matrix M - implementing paper formula 20"""
        old_dims = self._last_active_dims  # A_old
        new_dims = self.active_dims  # A_new

        old_size = len(old_dims)
        new_size = len(new_dims)

        # Create new M matrix
        M_new = np.zeros((new_size, new_size))

        # Calculate normalized importance scores
        if hasattr(self, 'dim_scores'):
            all_active_scores = self.dim_scores[self.active_dims].copy()
            all_active_scores = np.clip(all_active_scores, 1e-3, None)
            mean_score = np.mean(all_active_scores)
            normalized_mi = all_active_scores / mean_score
        else:
            normalized_mi = np.ones(new_size)

        # Calculate set relationships
        old_set = set(old_dims)
        new_set = set(new_dims)
        intersection = old_set & new_set  # A_old ∩ A_new (retained dimensions)
        new_only = new_set - old_set  # A_new \ A_old (new dimensions)

        print(f"Incremental update analysis:")
        print(f"  A_old: {old_dims}")
        print(f"  A_new: {new_dims}")
        print(f"  Retained (A_old ∩ A_new): {sorted(intersection)}")
        print(f"  New (A_new \\ A_old): {sorted(new_only)}")
        print(f"  Removed (A_old \\ A_new): {sorted(old_set - new_set)}")

        # Implement paper formula 20 general logic
        for i in range(new_size):
            for j in range(new_size):
                new_dim_i = new_dims[i]
                new_dim_j = new_dims[j]

                if new_dim_i in intersection and new_dim_j in intersection:
                    # Formula 20 part 1: M_new[i,j] = M_old[map(i), map(j)]
                    # Retained dimensions: directly copy corresponding values from old matrix
                    old_i = old_dims.index(new_dim_i)
                    old_j = old_dims.index(new_dim_j)
                    M_new[i, j] = self.M[-1][old_i, old_j]

                elif i == j and new_dim_i in new_only:
                    # Formula 20 part 2: m̄ · Ŝ(xᵢ) · δᵢⱼ (diagonal elements)
                    # New dimension diagonal elements
                    if old_size > 0:
                        m_bar = np.mean(np.diag(self.M[-1]))  # Average of old diagonal elements
                    else:
                        m_bar = 1.0
                    M_new[i, j] = m_bar * normalized_mi[i]

                # Set retained dimension diagonal elements to 0 first, then update with new MI scores
                elif i == j and new_dim_i in intersection:
                    M_new[i, j] = normalized_mi[i]

                # Off-diagonal elements involving new dimensions are temporarily set to 0

        # For new dimensions, recalculate off-diagonal elements with other dimensions
        if len(self.X) > 10:
            X_array = np.array(self.X)
            for i in range(new_size):
                for j in range(i + 1, new_size):
                    new_dim_i = new_dims[i]
                    new_dim_j = new_dims[j]

                    # If any dimension is new, recalculate off-diagonal elements
                    if new_dim_i in new_only or new_dim_j in new_only:
                        try:
                            rho_ij = abs(np.corrcoef(X_array[:, new_dim_i],
                                                     X_array[:, new_dim_j])[0, 1])
                        except:
                            rho_ij = 0.0

                        sigma_ij = 1.0 - abs(normalized_mi[i] - normalized_mi[j]) / max(normalized_mi[i],
                                                                                        normalized_mi[j])

                        # Paper formula 18: Mᵢⱼ = √(MᵢᵢMⱼⱼ) · (1 - ρᵢⱼ · σᵢⱼ)
                        interaction_factor = 1 - rho_ij * sigma_ij
                        M_ij = interaction_factor * np.sqrt(M_new[i, i] * M_new[j, j])
                        M_new[i, j] = M_new[j, i] = M_ij

        # Numerical stability
        M_new = (M_new + M_new.T) / 2
        M_new += 1e-6 * np.eye(new_size)

        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(M_new)
        if np.min(eigenvals) <= 1e-6:
            M_new += (1e-6 - np.min(eigenvals)) * np.eye(new_size)

        self.M[-1] = M_new

        # Update statistics
        reused_elements = len(intersection) * len(intersection)
        new_elements = new_size * new_size - reused_elements

        print(f"Incremental update statistics:")
        print(f"  Matrix size: {old_size}×{old_size} → {new_size}×{new_size}")
        print(
            f"  Reused elements: {reused_elements}/{new_size * new_size} ({reused_elements / (new_size * new_size) * 100:.1f}%)")
        print(f"  New elements: {new_elements}")

    def update_kernel_matrices(self):
        """Update kernel matrices - implementing paper formula 20 incremental update strategy"""
        min_samples_for_optimization = max(10, self.current_active_dim * 3)

        if len(self.X) < min_samples_for_optimization:
            print(
                f"Insufficient data points ({len(self.X)}/{min_samples_for_optimization}), skipping kernel matrix optimization")
            self._model_outdated = False
            return

        # Check if incremental update should be used
        if hasattr(self, '_last_active_dims') and self._should_use_incremental_update():
            # Implement paper formula 20 incremental update strategy
            print("Using incremental update strategy (paper formula 20)")
            self._incremental_update_M_matrix()
        else:
            # Complete reconstruction
            print("Rebuilding metric matrix (paper formulas: Mᵢᵢ = Ŝ(xᵢ), Mᵢⱼ = √(MᵢᵢMⱼⱼ)·(1-ρᵢⱼ·σᵢⱼ))")
            self._reinitialize_kernel_matrices()

        self._last_active_dims = self.active_dims.copy()
        self._model_outdated = False

    def _should_use_incremental_update(self):
        """Determine if incremental update should be used - formula 20 is general"""
        if not hasattr(self, '_last_active_dims'):
            return False

        if len(self._last_active_dims) == 0:
            return False

        # If new and old dimensions are identical, no update needed
        if set(self._last_active_dims) == set(self.active_dims):
            return False

        # All other cases can use incremental update
        old_set = set(self._last_active_dims)
        new_set = set(self.active_dims)

        intersection = old_set & new_set
        added = new_set - old_set
        removed = old_set - new_set

        print(f"Dimension change analysis:")
        print(f"  Retained dimensions: {sorted(intersection)} (total {len(intersection)})")
        print(f"  Added dimensions: {sorted(added)} (total {len(added)})")
        print(f"  Removed dimensions: {sorted(removed)} (total {len(removed)})")
        print(f"  → Using incremental update strategy (paper formula 20)")

        return True

    def expand_dimension(self):
        """Dimension expansion based on marginal information gain"""
        if self.current_active_dim >= self.total_dim:
            return False

        # Record dimension state before expansion (for incremental update)
        if not hasattr(self, '_last_active_dims'):
            self._last_active_dims = []
        self._last_active_dims = self.active_dims.copy()

        delta_d = self._calculate_delta_d()
        unused_dims = [d for d in range(self.total_dim) if d not in self.active_dims]

        if not unused_dims:
            return False

        marginal_gains = self._calculate_marginal_information_gain(unused_dims)
        sorted_dims = sorted(unused_dims, key=lambda d: marginal_gains.get(d, 0), reverse=True)

        n_expand = min(delta_d, len(sorted_dims))
        new_dims = sorted_dims[:n_expand]

        print(f"Expanding dimensions: adding {new_dims} (delta_d calculated: {delta_d})")
        for dim in new_dims:
            print(f"  Dimension {dim} marginal gain: {marginal_gains[dim]:.4f}")

        self.active_dims.extend(new_dims)
        self.current_active_dim += len(new_dims)

        # Mark matrices for update (will trigger incremental update)
        self._model_outdated = True
        self._data_outdated = True

        return True

    def optimize(self, n_iter=100):
        """Standard Bayesian optimization - fixed Ir handling logic"""
        self.max_iterations = n_iter
        best_values = []

        for i in range(n_iter):
            self.curr_iter = i

            x_next = self._select_next_point()
            outputs = self.objective_function(x_next)

            if isinstance(outputs, torch.Tensor):
                outputs = outputs.numpy()

            # Ensure correct output format
            if len(outputs.shape) == 1:
                outputs = outputs.reshape(1, -1)

            self.raw_y.append(outputs)
            y_next = outputs

            self.X.append(x_next)
            self.y.append(y_next)

            # Standard BO practice: mark cache as outdated, but don't rebuild immediately
            self._model_outdated = True
            self._data_outdated = True

            # Update best solution
            if self._is_feasible(y_next):
                obj_idx = 1
                current = y_next[0, obj_idx]
                if current < self.best_feasible_current:
                    self.best_feasible_current = current
                    self.best_feasible_point = x_next.copy()
                    print(f"\nNew best solution (iteration {i + 1}):")

            self._update_pareto_set()

            if hasattr(self, 'best_feasible_current'):
                best_values.append(self.best_feasible_current)
            else:
                best_values.append(float('inf'))

            # Update MI scores (for information ratio calculation)
            self._update_mi_scores()

            # Calculate information ratio and decide dimension management
            Ir, marginal_gains = self._calculate_information_ratio()

            if Ir > 1.0:
                print(
                    f"\nInformation ratio Ir = {Ir:.3f} > 1, triggering dimension management update (iteration {i + 1})")

                # Perform dimension management
                dimension_changed = False
                if self.expand_dimension():
                    print(f"Dimension expansion complete")
                    dimension_changed = True

                # Reordering may also change dimensions
                old_dims = self.active_dims.copy()
                self._reorder_dimensions()
                if set(old_dims) != set(self.active_dims):
                    dimension_changed = True
                    print(f"Dimension reordering complete")

                # Only rebuild kernel matrices when dimensions actually changed
                if dimension_changed:
                    self.update_kernel_matrices()
                    self._update_normalization_params()  # Recalculate normalization parameters
                    print(f"Dimension management complete, current active dimensions: {self.active_dims}")
                else:
                    print("No dimension changes, skipping kernel matrix update")

            else:
                # Ir ≤ 1: maintain current dimensions, perform standard BO update
                print(
                    f"Information ratio Ir = {Ir:.3f} ≤ 1, maintaining current dimensions {self.active_dims}, performing standard BO update")
                # Don't rebuild M matrix, GP will auto-update on next predict
                # Only update normalization parameters (due to new data point)
                self._update_normalization_params()

    def _is_feasible(self, y):
        """Check constraint satisfaction"""
        if not self.constraints:
            return True
        return (y[0, 0] >= self.constraints['gain'] and
                y[0, 2] >= self.constraints['phase'] and
                y[0, 3] >= self.constraints['gbw'])

    def _lhs_samples(self, n_samples):
        """Latin Hypercube Sampling"""
        current_seed = self.seed + len(self.X) + getattr(self, '_lhs_counter', 0)
        self._lhs_counter = getattr(self, '_lhs_counter', 0) + 1

        rng = np.random.RandomState(seed=current_seed)
        samples = np.zeros((n_samples, self.total_dim))

        for dim in range(self.total_dim):
            intervals = np.linspace(0, 1, n_samples + 1)
            for i in range(n_samples):
                samples[i, dim] = rng.uniform(intervals[i], intervals[i + 1])
            rng.shuffle(samples[:, dim])

        scaled_samples = np.zeros_like(samples)
        for i, (low, high) in enumerate(self.bounds):
            if high > low:  # Normal boundary range
                scaled_samples[:, i] = samples[:, i] * (high - low) + low
            else:
                # Handle abnormal boundaries
                print(f"Warning: abnormal boundaries for dimension {i} [{low}, {high}], using midpoint")
                scaled_samples[:, i] = (low + high) / 2

        return scaled_samples

    def _adaptive_initialize(self, initial_samples=10):
        """Initialization sampling"""
        print(f"Initialization parameters: total_dim={self.total_dim}, bounds length={len(self.bounds)}")

        X = []
        y_full = []
        feasible_count = 0

        all_points = self._lhs_samples(initial_samples)
        for x in all_points:
            result = self.objective_function(x)
            if isinstance(result, torch.Tensor):
                result = result.numpy()

            # Ensure correct output format
            if len(result.shape) == 1:
                result = result.reshape(1, -1)

            X.append(x)
            y_full.append(result)

            if self._is_feasible(result):
                feasible_count += 1
                obj_idx = 1
                current = result[0, obj_idx]
                if current < self.best_feasible_current:
                    self.best_feasible_current = current
                    self.best_feasible_point = x.copy()
                    print(f"New best feasible solution: Current={current:.2e}")

        print(
            f"Initialization complete: total samples={len(X)}, feasible solutions={feasible_count}, feasibility rate={feasible_count / len(X):.2%}")

        # Calculate MI scores
        X_array = np.array(X)
        y_array = np.array(y_full)
        if len(y_array.shape) == 3:
            y_array = y_array.reshape(len(y_array), -1)

        gain_count = phase_count = gbw_count = 0
        for y in y_full:
            if isinstance(y, np.ndarray) and y.shape[1] >= 4:
                if y[0, 0] >= self.constraints.get('gain', 0):
                    gain_count += 1
                if y[0, 2] >= self.constraints.get('phase', 0):
                    phase_count += 1
                if y[0, 3] >= self.constraints.get('gbw', 0):
                    gbw_count += 1

        scores = calculate_scores(
            dbx=X_array, dby=y_array, gain_num=gain_count, GBW_num=gbw_count,
            phase_num=phase_count, iter=0, init_num=initial_samples, input_dim=self.total_dim
        )

        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()

        self.X = X
        self.y = y_full
        self.raw_y = y_full
        self.dim_scores = scores

    def kernel_active(self, x1_active, x2_full):
        """Kernel function calculation

        Paper formula 15: k(x, x') = θ₀ exp(-1/2 (x - x')ᵀ M (x - x'))
        where x is the sub-vector corresponding to active_dims
        """
        # x1_active is already active_dims vector
        # x2_full is complete vector, need to extract active_dims
        x2_active = np.array([x2_full[i] for i in self.active_dims])

        x1_std = (x1_active - self.x_mean) / self.x_std
        x2_std = (x2_active - self.x_mean) / self.x_std

        diff = x1_std - x2_std

        # Paper formula 15: use metric matrix M to calculate distance
        scaled_diff = diff.T @ self.M[-1] @ diff
        scaled_diff = np.clip(scaled_diff, 0.0, 50.0)
        result = self.sigma * np.exp(-0.5 * scaled_diff)

        return result

    def expected_improvement(self, x_active, xi=0.01):
        """Constrained expected improvement acquisition function"""
        mu, sigma = self.predict(x_active)
        sigma = max(sigma, 1e-9)

        obj_idx = 1
        feasible_vals = [y[0, obj_idx] for y in self.y if self._is_feasible(y)]

        if feasible_vals:
            best_y = min(feasible_vals)
        else:
            best_y = min(y[0, 1] for y in self.y)

        imp = best_y - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = max(0.0, ei)

        # Constraint satisfaction probability
        constraint_satisfaction_probability = 1.0
        if self.constraints:
            constraints = ['gain', 'phase', 'gbw']
            for constraint in constraints:
                prob = self._predict_constraint_probability(x_active, constraint)
                constraint_satisfaction_probability *= prob

        cei = ei * constraint_satisfaction_probability
        return -cei

    def _predict_constraint_probability(self, x, constraint_name):
        """Predict constraint satisfaction probability"""
        constraint_info = self._get_constraint_info(constraint_name)
        if not constraint_info:
            return 0.5

        constraint_idx, constraint_direction, threshold = constraint_info

        # Use all samples, consistent with predict function
        indices = list(range(len(self.X)))

        X_train = []
        y_train = []

        for idx in indices:
            x_active = [self.X[idx][i] for i in self.active_dims]
            X_train.append(x_active)

            y_val = self.y[idx][0, constraint_idx]
            y_train.append(y_val)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Data preprocessing
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) or 1.0
        y_train_norm = (y_train - y_mean) / y_std

        # Use more stable kernel configuration
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2,
                normalize_y=False, alpha=1e-6
            )
            gp.fit(X_train, y_train_norm)

        # Prediction
        x_active = np.array([x[i] for i in self.active_dims]).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_c_norm, sigma_c_norm = gp.predict(x_active, return_std=True)

        # De-normalize
        mu_c = mu_c_norm[0] * y_std + y_mean
        sigma_c = sigma_c_norm[0] * y_std

        # Calculate constraint satisfaction probability
        if constraint_direction > 0:
            # Constraint form: f(x) >= threshold
            prob = 1 - norm.cdf((threshold - mu_c) / max(sigma_c, 1e-6))
        else:
            # Constraint form: f(x) <= threshold
            prob = norm.cdf((threshold - mu_c) / max(sigma_c, 1e-6))

        # Numerical stability: limit probability to reasonable range
        prob = max(min(prob, 0.99), 0.01)

        return prob

    def _get_constraint_info(self, constraint_name):
        """Get constraint information"""
        if constraint_name == 'gain':
            return 0, 1, self.constraints['gain']
        elif constraint_name == 'phase':
            return 2, 1, self.constraints['phase']
        elif constraint_name == 'gbw':
            return 3, 1, self.constraints['gbw']
        return None

    def predict(self, x_active):
        """GP prediction - unified sample count handling, remove limitations"""
        if not self.X or not self.y:
            return 0, 1e6

        # Confirm input dimensions
        if len(x_active) != len(self.active_dims):
            raise ValueError(
                f"Input dimension {len(x_active)} does not match active dimensions {len(self.active_dims)}")

        # Use all data, consistent with _update_mi_scores
        if not hasattr(self, '_train_indices') or self._data_outdated:
            self._train_indices = np.arange(len(self.X))  # Use all samples

        # Update kernel matrix when needed
        if not hasattr(self, '_K_cache') or self._model_outdated:
            indices = self._train_indices
            X_train = [self.X[i] for i in indices]
            self._K_cache = self.kernel_matrix(X_train, X_train)

            try:
                noise_diag = self.noise * np.eye(len(self._K_cache))
                self._K_noise_cache = self._K_cache + noise_diag
                self._L_cache = np.linalg.cholesky(self._K_noise_cache)
                self._chol_failed = False
            except np.linalg.LinAlgError:
                noise_diag = self.noise * 10 * np.eye(len(self._K_cache))
                self._K_noise_cache = self._K_cache + noise_diag
                try:
                    self._L_cache = np.linalg.cholesky(self._K_noise_cache)
                    self._chol_failed = False
                except:
                    self._chol_failed = True

        # Calculate k* vector
        k_star = np.array([self.kernel_active(x_active, self.X[i]) for i in self._train_indices])
        y_values = np.array([self.y[i][0, 1] for i in self._train_indices])

        try:
            if hasattr(self, '_chol_failed') and self._chol_failed:
                raise np.linalg.LinAlgError("Use backup strategy")

            alpha = cho_solve((self._L_cache, True), y_values)
            mu = k_star.T @ alpha
            v = solve_triangular(self._L_cache, k_star, lower=True)
            var = max(self.sigma - v.T @ v, 1e-6)
            result = (mu, np.sqrt(var))

        except:
            # Backup strategy: distance-based weighted prediction
            X_active_data = np.array([[self.X[i][j] for j in self.active_dims] for i in self._train_indices])

            dists = np.sum((X_active_data - x_active) ** 2, axis=1)
            weights = np.exp(-0.5 * dists / np.median(dists + 1e-10))
            weights = weights / (np.sum(weights) + 1e-10)

            mu = np.sum(weights * y_values)
            var = max(np.sum(weights * (y_values - mu) ** 2), self.sigma * 0.1)
            result = (mu, np.sqrt(var))

        return result

    def kernel_matrix(self, X1, X2):
        """Kernel matrix calculation - fixed to use direct matrix multiplication"""
        X1_active = np.array([[x[i] for i in self.active_dims] for x in X1], dtype=np.float64)
        X2_active = np.array([[x[i] for i in self.active_dims] for x in X2], dtype=np.float64)

        # Normalization
        if not hasattr(self, '_norm_params_cache') or self._data_outdated:
            self._norm_params_cache = {
                'x_mean': self.x_mean.astype(np.float64),
                'x_std': self.x_std.astype(np.float64),
                'M': self.M[-1].astype(np.float64)
            }

        X1_std = (X1_active - self._norm_params_cache['x_mean']) / self._norm_params_cache['x_std']
        X2_std = (X2_active - self._norm_params_cache['x_mean']) / self._norm_params_cache['x_std']

        # Use block computation to avoid memory issues
        block_size = min(500, max(50, 5000 // max(1, len(self.active_dims))))
        K = np.zeros((X1_std.shape[0], X2_std.shape[0]), dtype=np.float64)

        M = self._norm_params_cache['M']

        for i in range(0, X1_std.shape[0], block_size):
            i_end = min(i + block_size, X1_std.shape[0])
            X1_block = X1_std[i:i_end]

            for j in range(0, X2_std.shape[0], block_size):
                j_end = min(j + block_size, X2_std.shape[0])
                X2_block = X2_std[j:j_end]

                # Direct Mahalanobis distance calculation
                K_block = np.zeros((X1_block.shape[0], X2_block.shape[0]))

                for ii in range(X1_block.shape[0]):
                    for jj in range(X2_block.shape[0]):
                        diff = X1_block[ii] - X2_block[jj]
                        # Calculate Mahalanobis distance squared: diff^T M diff
                        mahal_dist_sq = diff.T @ M @ diff
                        mahal_dist_sq = np.clip(mahal_dist_sq, 0, 50)  # Numerical stability
                        K_block[ii, jj] = self.sigma * np.exp(-0.5 * mahal_dist_sq)

                K[i:i_end, j:j_end] = K_block

        return K

    def _update_pareto_set(self):
        """Maintain Pareto set"""
        if not hasattr(self, 'pareto_set'):
            self.pareto_set = []

        if len(self.X) == 0:
            return

        feasible_solutions = []
        for i, (x, y) in enumerate(zip(self.X, self.y)):
            if self._is_feasible(y):
                obj_value = y[0, 1]
                feasible_solutions.append({
                    'x': x.copy(),
                    'objective': obj_value,
                    'index': i
                })

        if len(feasible_solutions) == 0:
            self.pareto_set = []
            return

        feasible_solutions.sort(key=lambda sol: sol['objective'])
        n_keep = max(1, min(10, len(feasible_solutions) // 10))

        self.pareto_set = []
        for i in range(n_keep):
            self.pareto_set.append(feasible_solutions[i]['x'].copy())

        print(f"Pareto set updated, contains {len(self.pareto_set)} solutions")

    def _select_pareto_solution_for_inactive_params(self):
        """Select solution from Pareto set for inactive parameters"""
        if not hasattr(self, 'pareto_set') or len(self.pareto_set) == 0:
            if hasattr(self, 'best_feasible_point') and self.best_feasible_point is not None:
                return self.best_feasible_point.copy()
            else:
                return np.array([(self.bounds[i][0] + self.bounds[i][1]) / 2
                                 for i in range(self.total_dim)])

        rng = np.random.RandomState(seed=self.seed + len(self.X))
        selected_solution = rng.choice(len(self.pareto_set))
        return self.pareto_set[selected_solution].copy()

    def _select_next_point(self):
        """Select next optimization point"""

        def objective(x_active):
            # x_active is vector on active_dims, use directly
            try:
                eic_value = self.expected_improvement(x_active)
                return eic_value
            except:
                return 1e10

        bounds_active = []
        for dim_idx in self.active_dims:
            bounds_active.append((self.bounds[dim_idx][0], self.bounds[dim_idx][1]))

        if not self.active_dims:
            return self._lhs_samples(1)[0]

        best_x_active = None
        best_eic = float('inf')

        rng = np.random.RandomState(seed=self.seed + self.curr_iter)

        for restart in range(10):
            x0_active = []
            for dim_idx in self.active_dims:
                low, high = self.bounds[dim_idx]
                x0_active.append(rng.uniform(low, high))
            x0_active = np.array(x0_active)

            try:
                result = minimize(
                    objective, x0_active, method='L-BFGS-B', bounds=bounds_active,
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
                )

                if result.success and result.fun < best_eic:
                    best_eic = result.fun
                    best_x_active = result.x.copy()

            except Exception as e:
                continue

        if best_x_active is None:
            print("Warning: EIC optimization failed, using random sampling")
            return self._lhs_samples(1)[0]

        # Only build complete solution vector at the end (only needed here)
        x_next = np.zeros(self.total_dim)

        # Set active_dims values
        for i, dim_idx in enumerate(self.active_dims):
            x_next[dim_idx] = best_x_active[i]

        # Paper formula 21: use Pareto set to set inactive dimensions
        pareto_solution = self._select_pareto_solution_for_inactive_params()
        rng = np.random.RandomState(seed=self.seed + self.curr_iter)

        for i in range(self.total_dim):
            if i not in self.active_dims:
                x_plus_i = pareto_solution[i]
                noise_scale = (self.bounds[i][1] - self.bounds[i][0]) * 1e-5
                epsilon_i = rng.normal(0, noise_scale)
                x_next[i] = x_plus_i + epsilon_i
                x_next[i] = np.clip(x_next[i], self.bounds[i][0], self.bounds[i][1])

        return x_next
