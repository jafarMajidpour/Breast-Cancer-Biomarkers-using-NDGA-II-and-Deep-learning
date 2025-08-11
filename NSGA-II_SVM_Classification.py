import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, mean_squared_error,
                           classification_report, confusion_matrix)
from sklearn.impute import SimpleImputer
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NSGAIIFeatureSelector:
    def __init__(self, 
                 classifier_type='rf',
                 population_size=50,
                 generations=30,
                 crossover_prob=0.8,
                 mutation_prob=0.2,
                 random_state=42):
        """
        Initialize NSGA-II Feature Selector
        
        Args:
            classifier_type: 'rf' for Random Forest or 'svm' for SVM
            population_size: Size of the population for NSGA-II
            generations: Number of generations for evolution
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            random_state: Random state for reproducibility
        """
        self.classifier_type = classifier_type
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None
        
        # NSGA-II setup
        self.toolbox = None
        self.best_features = None
        self.best_performance = None
        
        # Set random seeds
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def load_and_prepare_data(self, csv_path, target_column, test_size=0.2):
        """
        Load CSV data and prepare for training
        
        Args:
            csv_path: Path to CSV file
            target_column: Name of the target column
            test_size: Proportion of data for testing
        """
        print(f"Loading data from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")
        
        # Remove non-feature columns (case_name, etc.)
        feature_columns = [col for col in df.columns 
                          if col != target_column and 
                          not col.lower().startswith('case') and
                          not col.lower().startswith('id')]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        self.feature_names = feature_columns
        print(f"Features: {len(self.feature_names)}")
        print(f"Target classes: {y.value_counts().to_dict()}")
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Encode target if categorical
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, 
            stratify=y_encoded
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
    
    def create_classifier(self):
        """Create classifier based on specified type"""
        if self.classifier_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.classifier_type == 'svm':
            return SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True  # Enable probability estimates for AUC
            )
        else:
            raise ValueError("classifier_type must be 'rf' or 'svm'")
    
    def evaluate_feature_subset(self, feature_mask):
        """
        Evaluate a feature subset using cross-validation
        
        Args:
            feature_mask: Binary array indicating selected features
            
        Returns:
            Tuple of (negative_f1_score, num_features) for minimization
        """
        # Select features
        selected_indices = np.where(feature_mask)[0]
        
        if len(selected_indices) == 0:
            return (1.0, len(self.feature_names))  # Worst case
        
        X_selected = self.X_train[:, selected_indices]
        
        # Create classifier
        classifier = self.create_classifier()
        
        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(
                classifier, X_selected, self.y_train,
                cv=cv,
                scoring=['f1_weighted', 'accuracy'],
                n_jobs=-1,
                error_score='raise'
            )
            
            # Calculate mean performance
            mean_f1 = np.mean(cv_results['test_f1_weighted'])
            
            # Return negative F1 for minimization and number of features
            return (-mean_f1, len(selected_indices))
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return (1.0, len(selected_indices))
    
    def setup_nsga2(self):
        """Setup NSGA-II algorithm using DEAP"""
        print("Setting up NSGA-II algorithm...")
        
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Attribute generator (binary for feature selection)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Structure initializers
        n_features = len(self.feature_names)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_bool, n_features)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        # Evaluation function
        self.toolbox.register("evaluate", self.evaluate_feature_subset)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)
    
    def run_nsga2_optimization(self):
        """
        Run NSGA-II optimization for feature selection
        
        Returns:
            Pareto front of solutions
        """
        print(f"Running NSGA-II optimization...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Hall of Fame to keep track of best individuals
        hof = tools.ParetoFront()
        
        # Run evolution
        population, logbook = algorithms.eaMuPlusLambda(
            population, self.toolbox, 
            mu=self.population_size, 
            lambda_=self.population_size,
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        print(f"Optimization completed. Pareto front size: {len(hof)}")
        
        return hof, logbook
    
    def select_best_solution(self, pareto_front, preference='balanced'):
        """
        Select best solution from Pareto front
        
        Args:
            pareto_front: Pareto front from NSGA-II
            preference: 'performance', 'minimal', or 'balanced'
            
        Returns:
            Best feature mask
        """
        if not pareto_front:
            raise ValueError("Empty Pareto front")
        
        solutions = []
        for individual in pareto_front:
            neg_f1, n_features = individual.fitness.values
            f1_score = -neg_f1
            solutions.append({
                'individual': individual,
                'f1_score': f1_score,
                'n_features': n_features,
                'feature_ratio': n_features / len(self.feature_names)
            })
        
        solutions_df = pd.DataFrame(solutions)
        
        if preference == 'performance':
            # Select solution with highest F1 score
            best_idx = solutions_df['f1_score'].idxmax()
        elif preference == 'minimal':
            # Select solution with fewest features among top performers
            top_performers = solutions_df[solutions_df['f1_score'] >= solutions_df['f1_score'].quantile(0.9)]
            best_idx = top_performers['n_features'].idxmin()
        else:  # balanced
            # Balance between performance and feature count
            solutions_df['score'] = (solutions_df['f1_score'] / solutions_df['f1_score'].max() + 
                                   (1 - solutions_df['feature_ratio'])) / 2
            best_idx = solutions_df['score'].idxmax()
        
        best_solution = solutions_df.iloc[best_idx]
        self.best_features = np.array(best_solution['individual'], dtype=bool)
        
        print(f"Selected solution:")
        print(f"  F1 Score: {best_solution['f1_score']:.4f}")
        print(f"  Features: {best_solution['n_features']}/{len(self.feature_names)}")
        print(f"  Feature ratio: {best_solution['feature_ratio']:.3f}")
        
        return self.best_features
    
    def evaluate_final_model(self):
        """
        Evaluate the final model with selected features on test set
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_features is None:
            raise ValueError("No features selected. Run optimization first.")
        
        # Select features
        selected_indices = np.where(self.best_features)[0]
        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]
        
        print(f"Final evaluation with {len(selected_indices)} selected features")
        
        # Train final model
        final_classifier = self.create_classifier()
        final_classifier.fit(X_train_selected, self.y_train)
        
        # Predictions
        y_pred = final_classifier.predict(X_test_selected)
        y_pred_proba = final_classifier.predict_proba(X_test_selected)
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        metrics['precision'] = precision_score(self.y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(self.y_test, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(self.y_test, y_pred, average='weighted')
        
        # MSE (for continuous representation of classification)
        metrics['mse'] = mean_squared_error(self.y_test, y_pred)
        
        # AUC (multiclass handling)
        if len(np.unique(self.y_test)) == 2:
            # Binary classification
            metrics['auc'] = roc_auc_score(self.y_test, y_pred_proba[:, 1])
        else:
            # Multiclass classification
            metrics['auc'] = roc_auc_score(self.y_test, y_pred_proba, 
                                         multi_class='ovr', average='weighted')
        
        # Cross-validation metrics
        print("Performing 5-fold cross-validation on selected features...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_validate(
            self.create_classifier(), 
            X_train_selected, 
            self.y_train,
            cv=cv,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 
                    'f1_weighted', 'roc_auc_ovr_weighted'],
            n_jobs=-1
        )
        
        # Add CV metrics
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 
                      'f1_weighted', 'roc_auc_ovr_weighted']:
            metrics[f'cv_{metric}_mean'] = np.mean(cv_scores[f'test_{metric}'])
            metrics[f'cv_{metric}_std'] = np.std(cv_scores[f'test_{metric}'])
        
        # Store results
        self.best_performance = metrics
        
        return metrics, final_classifier, selected_indices

class MRIClassificationPipeline:
    def __init__(self, csv_path, target_column, classifier_type='rf'):
        """
        Initialize the complete MRI classification pipeline
        
        Args:
            csv_path: Path to CSV file containing features
            target_column: Name of target column for classification
            classifier_type: 'rf' for Random Forest or 'svm' for SVM
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.classifier_type = classifier_type
        self.selector = None
        self.results = {}
    
    def run_pipeline(self, 
                    population_size=50, 
                    generations=20,
                    test_size=0.2,
                    selection_preference='balanced'):
        """
        Run the complete pipeline
        
        Args:
            population_size: NSGA-II population size
            generations: Number of generations
            test_size: Test set proportion
            selection_preference: 'performance', 'minimal', or 'balanced'
        """
        print("=" * 70)
        print("MRI CLASSIFICATION PIPELINE WITH NSGA-II FEATURE SELECTION")
        print("=" * 70)
        
        # Initialize feature selector
        self.selector = NSGAIIFeatureSelector(
            classifier_type=self.classifier_type,
            population_size=population_size,
            generations=generations
        )
        
        # Step 1: Load and prepare data
        print("\n1. LOADING AND PREPARING DATA")
        print("-" * 40)
        self.selector.load_and_prepare_data(self.csv_path, self.target_column, test_size)
        
        # Step 2: Setup NSGA-II
        print("\n2. SETTING UP NSGA-II OPTIMIZATION")
        print("-" * 40)
        self.selector.setup_nsga2()
        
        # Step 3: Run optimization
        print("\n3. RUNNING NSGA-II FEATURE SELECTION")
        print("-" * 40)
        pareto_front, logbook = self.selector.run_nsga2_optimization()
        
        # Step 4: Select best solution
        print("\n4. SELECTING BEST SOLUTION")
        print("-" * 40)
        best_features = self.selector.select_best_solution(pareto_front, selection_preference)
        
        # Step 5: Final evaluation
        print("\n5. FINAL MODEL EVALUATION")
        print("-" * 40)
        metrics, final_model, selected_indices = self.selector.evaluate_final_model()
        
        # Store results
        self.results = {
            'metrics': metrics,
            'selected_features': best_features,
            'selected_feature_names': [self.selector.feature_names[i] for i in selected_indices],
            'selected_indices': selected_indices,
            'final_model': final_model,
            'pareto_front': pareto_front,
            'logbook': logbook
        }
        
        self.print_results()
        return self.results
    
    def print_results(self):
        """Print comprehensive results"""
        metrics = self.results['metrics']
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        
        print(f"Classifier: {self.classifier_type.upper()}")
        print(f"Selected Features: {len(self.results['selected_indices'])}/{len(self.selector.feature_names)}")
        
        print(f"\nTEST SET PERFORMANCE:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  MSE:       {metrics['mse']:.4f}")
        
        print(f"\n5-FOLD CROSS-VALIDATION RESULTS:")
        print(f"  Accuracy:  {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  Precision: {metrics['cv_precision_weighted_mean']:.4f} ± {metrics['cv_precision_weighted_std']:.4f}")
        print(f"  Recall:    {metrics['cv_recall_weighted_mean']:.4f} ± {metrics['cv_recall_weighted_std']:.4f}")
        print(f"  F1-Score:  {metrics['cv_f1_weighted_mean']:.4f} ± {metrics['cv_f1_weighted_std']:.4f}")
        print(f"  AUC:       {metrics['cv_roc_auc_ovr_weighted_mean']:.4f} ± {metrics['cv_roc_auc_ovr_weighted_std']:.4f}")
    
    def save_results(self, output_dir="results"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
        
        # Save selected features
        selected_features_df = pd.DataFrame({
            'feature_index': self.results['selected_indices'],
            'feature_name': self.results['selected_feature_names']
        })
        selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)
        
        # Save feature selection mask
        feature_mask_df = pd.DataFrame({
            'feature_name': self.selector.feature_names,
            'selected': self.results['selected_features']
        })
        feature_mask_df.to_csv(os.path.join(output_dir, 'feature_selection_mask.csv'), index=False)
        
        print(f"Results saved to {output_dir}/")
    
    def plot_pareto_front(self, save_path=None):
        """Plot the Pareto front"""
        try:
            pareto_front = self.results['pareto_front']
            
            # Extract objectives
            f1_scores = [-ind.fitness.values[0] for ind in pareto_front]
            n_features = [ind.fitness.values[1] for ind in pareto_front]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(n_features, f1_scores, alpha=0.7, s=60)
            plt.xlabel('Number of Features')
            plt.ylabel('F1 Score')
            plt.title('NSGA-II Pareto Front: Feature Selection Trade-off')
            plt.grid(True, alpha=0.3)
            
            # Highlight selected solution
            selected_idx = np.where(self.results['selected_features'])[0]
            selected_f1 = self.results['metrics']['f1_score']
            plt.scatter([len(selected_idx)], [selected_f1], 
                       color='red', s=100, marker='*', 
                       label='Selected Solution')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting Pareto front: {str(e)}")

def main():
    """Main function to run the complete pipeline"""
    
    # Configuration
    CSV_PATH = "E:/__Deep Learning/___Her2_ER_PR Paper/Radiomics.csv"  # UPDATE THIS PATH
    TARGET_COLUMN = "target"  # UPDATE THIS COLUMN NAME
    CLASSIFIER_TYPE = "rf"  # 'rf' for Random Forest, 'svm' for SVM
    
    # Pipeline parameters
    POPULATION_SIZE = 50
    GENERATIONS = 20
    TEST_SIZE = 0.2
    SELECTION_PREFERENCE = 'balanced'  # 'performance', 'minimal', or 'balanced'
    
    try:
        # Initialize and run pipeline
        pipeline = MRIClassificationPipeline(
            csv_path=CSV_PATH,
            target_column=TARGET_COLUMN,
            classifier_type=CLASSIFIER_TYPE
        )
        
        # Run the complete pipeline
        results = pipeline.run_pipeline(
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            test_size=TEST_SIZE,
            selection_preference=SELECTION_PREFERENCE
        )
        
        # Save results
        pipeline.save_results("results")
        
        # Plot Pareto front
        pipeline.plot_pareto_front("results/pareto_front.png")
        
        return pipeline, results
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return None, None

# Utility functions
def compare_classifiers(csv_path, target_column, selected_features=None):
    """
    Compare Random Forest and SVM performance
    
    Args:
        csv_path: Path to CSV file
        target_column: Target column name
        selected_features: Array of selected feature indices (optional)
    """
    print("Comparing Random Forest and SVM classifiers...")
    
    results_comparison = {}
    
    for classifier_type in ['rf', 'svm']:
        print(f"\nTesting {classifier_type.upper()}...")
        
        pipeline = MRIClassificationPipeline(csv_path, target_column, classifier_type)
        selector = NSGAIIFeatureSelector(classifier_type=classifier_type, generations=10)
        selector.load_and_prepare_data(csv_path, target_column)
        
        if selected_features is not None:
            # Use provided feature selection
            selector.best_features = selected_features
            metrics, _, _ = selector.evaluate_final_model()
        else:
            # Quick feature selection
            selector.setup_nsga2()
            pareto_front, _ = selector.run_nsga2_optimization()
            selector.select_best_solution(pareto_front)
            metrics, _, _ = selector.evaluate_final_model()
        
        results_comparison[classifier_type] = metrics
    
    # Print comparison
    print("\n" + "=" * 50)
    print("CLASSIFIER COMPARISON")
    print("=" * 50)
    
    comparison_df = pd.DataFrame(results_comparison).T
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc']])
    
    return results_comparison

def load_and_preview_csv(csv_path, max_rows=5):
    """
    Load and preview CSV file structure
    
    Args:
        csv_path: Path to CSV file
        max_rows: Maximum rows to display
    """
    try:
        df = pd.read_csv(csv_path)
        
        print(f"CSV Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:10])}...")
        print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        print(f"  Missing values: {missing}")
        
        # Show sample data
        print(f"\nFirst {max_rows} rows:")
        print(df.head(max_rows))
        
        # Show target distribution if identifiable
        potential_targets = [col for col in df.columns 
                           if col.lower() in ['target', 'label', 'class', 'diagnosis']]
        
        if potential_targets:
            target_col = potential_targets[0]
            print(f"\nTarget column '{target_col}' distribution:")
            print(df[target_col].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

if __name__ == "__main__":
    print("MRI Classification Pipeline with NSGA-II Feature Selection")
    print("=" * 70)
    
    # Example usage
    print("\nCONFIGURATION STEPS:")
    print("1. Update CSV_PATH to your feature file")
    print("2. Set TARGET_COLUMN to your target variable name")
    print("3. Choose CLASSIFIER_TYPE: 'rf' or 'svm'")
    print("4. Adjust NSGA-II parameters as needed")
    
    # Preview CSV structure (uncomment to use)
    # csv_file = "your_features.csv"  # Update this path
    # preview_df = load_and_preview_csv(csv_file)
    
    # Run pipeline (uncomment to use)
    # pipeline, results = main()
    
    # Compare classifiers (uncomment to use)
    # comparison = compare_classifiers("your_features.csv", "target")
    
    print("\nTo run the pipeline, uncomment the main() call and update the configuration variables.")

# Example of custom NSGA-II settings for different scenarios
def get_nsga2_configs():
    """
    Get different NSGA-II configuration presets
    
    Returns:
        Dictionary of configuration presets
    """
    configs = {
        'fast': {
            'population_size': 30,
            'generations': 10,
            'description': 'Quick optimization for testing'
        },
        'standard': {
            'population_size': 50,
            'generations': 20,
            'description': 'Standard optimization'
        },
        'thorough': {
            'population_size': 100,
            'generations': 50,
            'description': 'Thorough optimization (slow but better results)'
        },
        'large_dataset': {
            'population_size': 80,
            'generations': 30,
            'description': 'Optimized for large feature sets'
        }
    }
    
    return configs

# Feature importance analysis
def analyze_selected_features(pipeline_results, top_n=20):
    """
    Analyze the importance of selected features
    
    Args:
        pipeline_results: Results from the pipeline
        top_n: Number of top features to analyze
    """
    if 'final_model' not in pipeline_results:
        print("No trained model found in results")
        return
    
    model = pipeline_results['final_model']
    feature_names = pipeline_results['selected_feature_names']
    
    # Get feature importance (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"Top {top_n} most important selected features:")
        print(importance_df.head(top_n))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance_df['importance'].head(top_n))
        plt.yticks(range(top_n), importance_df['feature'].head(top_n))
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Feature importance not available for this classifier")
        return None