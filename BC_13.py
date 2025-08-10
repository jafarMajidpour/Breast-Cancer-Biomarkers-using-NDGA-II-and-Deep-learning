import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class BreastCancerBiomarkerDetector:
   
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.set_random_seeds()
        
        # Initialize components
        self.radiomics_extractor = None
        self.mobilenet_model = None
        self.pca_reducer = None
        self.scalers = {}
        self.feature_encoders = {}
        
        # NSGA-II parameters
        self.nsga_params = {
            'population_size': 150,
            'generations': 200,
            'crossover_prob': 0.7,
            'mutation_prob': 0.4,
            'mutation_rate': 0.1
        }
        
        # Initialize DEAP framework
        self.setup_nsga_ii()
        
    def set_random_seeds(self):  #Set random seeds for reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def setup_nsga_ii(self):    #Setup DEAP framework for NSGA-II optimization
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.nsga_params['mutation_rate'])
        self.toolbox.register("select", tools.selNSGA2)
    
    def extract_radiomics_features(self, image_path: str, mask_path: str) -> np.ndarray:

        if self.radiomics_extractor is None:
            # Initialize radiomics feature extractor
            self.radiomics_extractor = featureextractor.RadiomicsFeatureExtractor()
            
            # Configure extraction settings
            settings = {
                'binWidth': 25,
                'resampledPixelSpacing': None,
                'interpolator': sitk.sitkBSpline,
                'enableCExtensions': True
            }
            self.radiomics_extractor.settings.update(settings)
            
            # Enable feature classes
            self.radiomics_extractor.enableFeatureClassByName('shape')
            self.radiomics_extractor.enableFeatureClassByName('firstorder')
            self.radiomics_extractor.enableFeatureClassByName('glcm')
            self.radiomics_extractor.enableFeatureClassByName('glrlm')
            self.radiomics_extractor.enableFeatureClassByName('glszm')
            self.radiomics_extractor.enableFeatureClassByName('ngtdm')
            self.radiomics_extractor.enableFeatureClassByName('gldm')
        
        # Load image and mask
        image_path = 'E:/__Deep Learning/___Her2_ER_PR Paper/Dataset/MRI/'
        mask_path = 'E:/__Deep Learning/___Her2_ER_PR Paper/Dataset/Mask/'
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
       
        # Extract features
        features = self.radiomics_extractor.execute(image, mask)
        
        # Convert to numpy array (exclude metadata)
        feature_vector = []
        for key, value in features.items():
            if not key.startswith('diagnostics_'):
                try:
                    feature_vector.append(float(value))
                except (ValueError, TypeError):
                    feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def setup_mobilenet_model(self):
        """Initialize MobileNetV2 model for deep feature extraction"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(224, 224, 3), pooling='avg')
        
        self.mobilenet_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output  # Global average pooling layer
        )
        
        # Fine-tune with small learning rate for medical imaging
        for layer in self.mobilenet_model.layers[:-10]:
            layer.trainable = False
    
    def extract_deep_features(self, image_slices: List[np.ndarray]) -> np.ndarray:

        if self.mobilenet_model is None:
            self.setup_mobilenet_model()
        
        slice_features = []
        
        for slice_img in image_slices:
            # Preprocess slice for MobileNetV2
            if len(slice_img.shape) == 2:
                # Convert grayscale to RGB
                slice_img = np.stack([slice_img] * 3, axis=-1)
            
            # Resize to 224x224
            slice_img = tf.image.resize(slice_img, [224, 224])
            slice_img = tf.cast(slice_img, tf.float32)
            
            # Normalize for MobileNetV2
            slice_img = preprocess_input(slice_img)
            slice_img = tf.expand_dims(slice_img, 0)
            
            # Extract features
            features = self.mobilenet_model(slice_img)
            slice_features.append(features.numpy().flatten())
        
        # Stack all slice features
        slice_features = np.array(slice_features)
        
        # Apply PCA for dimensionality reduction if not initialized
        if self.pca_reducer is None:
            self.pca_reducer = PCA(n_components=0.95, random_state=self.random_state)
            reduced_features = self.pca_reducer.fit_transform(slice_features)
        else:
            reduced_features = self.pca_reducer.transform(slice_features)
        
        # Aggregate across slices using max pooling
        aggregated_features = np.max(reduced_features, axis=0)
        
        return aggregated_features
    
    def prepare_features(self, radiomics_features: np.ndarray, 
                        deep_features: np.ndarray, 
                        demographic_features: np.ndarray) -> np.ndarray:

        # Standardize continuous features
        if 'radiomics' not in self.scalers:
            self.scalers['radiomics'] = StandardScaler()
            radiomics_scaled = self.scalers['radiomics'].fit_transform(radiomics_features)
        else:
            radiomics_scaled = self.scalers['radiomics'].transform(radiomics_features)
        
        if 'deep' not in self.scalers:
            self.scalers['deep'] = StandardScaler()
            deep_scaled = self.scalers['deep'].fit_transform(deep_features)
        else:
            deep_scaled = self.scalers['deep'].transform(deep_features)
        
        # Encode demographic features (assuming categorical)
        if 'demographic' not in self.feature_encoders:
            self.feature_encoders['demographic'] = OneHotEncoder(sparse_output=False, 
                                                               handle_unknown='ignore')
            demographic_encoded = self.feature_encoders['demographic'].fit_transform(demographic_features)
        else:
            demographic_encoded = self.feature_encoders['demographic'].transform(demographic_features)
        
        # Combine all features
        combined_features = np.hstack([radiomics_scaled, deep_scaled, demographic_encoded])
        
        return combined_features
    
    def evaluate_individual(self, individual: List[int], X: np.ndarray, y: np.ndarray, 
                          classifier_type: str = 'rf') -> Tuple[float, float]:

        # Select features based on individual
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        
        if len(selected_indices) == 0:
            return 1.0, 1.0  # Worst possible fitness
        
        X_selected = X[:, selected_indices]
        
        # Train classifier with selected features
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:  # svm
            clf = SVC(kernel='rbf', random_state=self.random_state)
        
        # Use stratified k-fold for evaluation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        mse_scores = []
        
        for train_idx, val_idx in skf.split(X_selected, y):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            
            # Calculate MSE
            mse = mean_squared_error(y_val, y_pred)
            mse_scores.append(mse)
        
        avg_mse = np.mean(mse_scores)
        feature_ratio = len(selected_indices) / len(individual)
        
        return feature_ratio, avg_mse
    
    def nsga_ii_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                 classifier_type: str = 'rf') -> List[int]:

        n_features = X.shape[1]
        
        # Register individual and evaluation function
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_bool, n_features)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual, 
                            X=X, y=y, classifier_type=classifier_type)
        
        # Initialize population
        population = self.toolbox.population(n=self.nsga_params['population_size'])
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution parameters
        CXPB, MUTPB = self.nsga_params['crossover_prob'], self.nsga_params['mutation_prob']
        NGEN = self.nsga_params['generations']
        
        # Store convergence data
        self.convergence_data = {'generations': [], 'min_mse': [], 'min_features': []}
        
        # Evolution loop
        for gen in range(NGEN):
            # Select offspring
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Apply crossover and mutation
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    self.toolbox.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
            
            for ind in offspring:
                if random.random() <= MUTPB:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            population = self.toolbox.select(population + offspring, 
                                           self.nsga_params['population_size'])
            
            # Track convergence
            fitnesses = [ind.fitness.values for ind in population]
            min_mse = min([fit[1] for fit in fitnesses])
            min_features = min([fit[0] for fit in fitnesses])
            
            self.convergence_data['generations'].append(gen)
            self.convergence_data['min_mse'].append(min_mse)
            self.convergence_data['min_features'].append(min_features)
            
            if gen % 50 == 0:
                print(f"Generation {gen}: Min MSE = {min_mse:.4f}, Min Feature Ratio = {min_features:.4f}")
        
        # Select best solution from Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Choose solution with minimum MSE
        best_individual = min(pareto_front, key=lambda x: x.fitness.values[1])
        
        return best_individual
    
    def nested_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                               biomarker_name: str = 'HER2') -> Dict[str, float]:
    
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        performance_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1_score': [], 'auc': [], 'mse': []
        }
        
        classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', probability=True, random_state=self.random_state)
        }
        
        fold = 0
        for train_outer_idx, test_outer_idx in outer_cv.split(X, y):
            fold += 1
            print(f"\nOuter fold {fold}/5 for {biomarker_name}")
            
            X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
            y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]
            
            best_score = 0
            best_features = None
            best_params = None
            best_classifier_type = None
            
            # Inner cross-validation for model selection
            for clf_name in ['rf', 'svm']:
                print(f"  Testing {clf_name} classifier...")
                
                # Feature selection on inner training data only
                best_individual = self.nsga_ii_feature_selection(
                    X_train_outer, y_train_outer, clf_name
                )
                
                # Get selected features
                selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
                X_train_selected = X_train_outer[:, selected_features]
                
                # Hyperparameter tuning
                if clf_name == 'rf':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'max_features': ['sqrt', 'log2']
                    }
                else:  # svm
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'gamma': ['scale', 'auto'],
                        'kernel': ['rbf', 'linear']
                    }
                
                grid_search = GridSearchCV(
                    classifiers[clf_name], param_grid, 
                    cv=inner_cv, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train_selected, y_train_outer)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_features = selected_features
                    best_params = grid_search.best_params_
                    best_classifier_type = clf_name
            
            # Train final model with best configuration
            X_final_train = X_train_outer[:, best_features]
            X_final_test = X_test_outer[:, best_features]
            
            final_classifier = classifiers[best_classifier_type]
            final_classifier.set_params(**best_params)
            final_classifier.fit(X_final_train, y_train_outer)
            
            # Predict on test set
            y_pred = final_classifier.predict(X_final_test)
            y_pred_proba = final_classifier.predict_proba(X_final_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_outer, y_pred)
            precision = precision_score(y_test_outer, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_outer, y_pred, average='weighted')
            f1 = f1_score(y_test_outer, y_pred, average='weighted')
            auc = roc_auc_score(y_test_outer, y_pred_proba)
            mse = mean_squared_error(y_test_outer, y_pred)
            
            # Store results
            performance_metrics['accuracy'].append(accuracy)
            performance_metrics['precision'].append(precision)
            performance_metrics['recall'].append(recall)
            performance_metrics['f1_score'].append(f1)
            performance_metrics['auc'].append(auc)
            performance_metrics['mse'].append(mse)
            
            print(f"  Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            print(f"  Selected {len(best_features)} features ({len(best_features)/X.shape[1]*100:.1f}%)")
        
        # Calculate mean and std for each metric
        results = {}
        for metric, values in performance_metrics.items():
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        return results
    
    def plot_convergence(self):
        """Plot NSGA-II convergence curves"""
        if not hasattr(self, 'convergence_data'):
            print("No convergence data available. Run feature selection first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE convergence
        ax1.plot(self.convergence_data['generations'], self.convergence_data['min_mse'])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Minimum MSE')
        ax1.set_title('MSE Convergence')
        ax1.grid(True)
        
        # Feature count convergence
        ax2.plot(self.convergence_data['generations'], self.convergence_data['min_features'])
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Minimum Feature Ratio')
        ax2.set_title('Feature Count Convergence')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_experiment(self, radiomics_features: np.ndarray, deep_features: np.ndarray,
                      demographic_features: np.ndarray, labels: Dict[str, np.ndarray]) -> Dict[str, Dict]:

        # Prepare combined features
        X_combined = self.prepare_features(radiomics_features, deep_features, demographic_features)
        
        print(f"Combined feature matrix shape: {X_combined.shape}")
        print(f"Radiomics features: {radiomics_features.shape[1]}")
        print(f"Deep features: {deep_features.shape[1]}")
        print(f"Demographic features: {demographic_features.shape[1]}")
        
        results = {}
        
        for biomarker, y in labels.items():
            print(f"\n{'='*60}")
            print(f"Processing {biomarker} biomarker")
            print(f"{'='*60}")
            
            # Run nested cross-validation
            biomarker_results = self.nested_cross_validation(X_combined, y, biomarker)
            results[biomarker] = biomarker_results
            
            # Print results
            print(f"\n{biomarker} Results:")
            print(f"Accuracy: {biomarker_results['accuracy_mean']:.4f} ± {biomarker_results['accuracy_std']:.4f}")
            print(f"Precision: {biomarker_results['precision_mean']:.4f} ± {biomarker_results['precision_std']:.4f}")
            print(f"Recall: {biomarker_results['recall_mean']:.4f} ± {biomarker_results['recall_std']:.4f}")
            print(f"F1-Score: {biomarker_results['f1_score_mean']:.4f} ± {biomarker_results['f1_score_std']:.4f}")
            print(f"AUC: {biomarker_results['auc_mean']:.4f} ± {biomarker_results['auc_std']:.4f}")
            print(f"MSE: {biomarker_results['mse_mean']:.4f} ± {biomarker_results['mse_std']:.4f}")
        
        return results

# Example usage and demonstration
def main():
    #Demonstration of the breast cancer biomarker detection framework
    
    print("Breast Cancer Biomarker Detection Framework")
    print("=" * 50)
    
    # Initialize the detector
    detector = BreastCancerBiomarkerDetector(random_state=42)
    
    # Example with synthetic data (replace with real data loading)
    print("Generating synthetic data for demonstration...")
    
    n_samples = 500
    
    # Synthetic radiomics features (529 features as mentioned in paper)
    radiomics_features = np.random.randn(n_samples, 529)
    
    # Synthetic deep features (after PCA reduction)
    deep_features = np.random.randn(n_samples, 320)  # Reduced dimensionality
    
    # Synthetic demographic features
    demographic_features = np.random.randint(0, 3, size=(n_samples, 3))
    
    # Synthetic biomarker labels
    labels = {
        'HER2': np.random.binomial(1, 0.18, n_samples),  # 18% positive as in paper
        'ER': np.random.binomial(1, 0.74, n_samples),    # 74% positive as in paper  
        'PR': np.random.binomial(1, 0.65, n_samples)     # 65% positive as in paper
    }
    
    print(f"Dataset statistics:")
    print(f"Total samples: {n_samples}")
    for biomarker, y in labels.items():
        pos_rate = np.mean(y) * 100
        print(f"{biomarker} positive rate: {pos_rate:.1f}%")
    
    # Run the complete experiment
    results = detector.run_experiment(
        radiomics_features, deep_features, demographic_features, labels
    )
    
    # Plot convergence (if available)
    if hasattr(detector, 'convergence_data'):
        detector.plot_convergence()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    
    # Summary table
    print("\nSummary Results:")
    print("-" * 80)
    print(f"{'Biomarker':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'AUC':<12}")
    print("-" * 80)
    
    for biomarker, result in results.items():
        print(f"{biomarker:<10} "
              f"{result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}  "
              f"{result['precision_mean']:.3f}±{result['precision_std']:.3f}  "
              f"{result['recall_mean']:.3f}±{result['recall_std']:.3f}  "
              f"{result['auc_mean']:.3f}±{result['auc_std']:.3f}")

if __name__ == "__main__":
    main()