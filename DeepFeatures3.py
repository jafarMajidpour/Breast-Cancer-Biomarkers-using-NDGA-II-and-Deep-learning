import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
from tqdm import tqdm
import glob
from pathlib import Path

class MRIFeatureExtractor:
    def __init__(self, target_size=(224, 224), n_components=1280):

        self.target_size = target_size
        self.n_components = n_components
        self.model = None
        self.pca = None
        self.scaler = None
        
    def load_mobilenetv2(self):  #Load MobileNetV2 model 
        print("Loading MobileNetV2 model...")
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.target_size, 3),
            pooling='avg'  # Global average pooling to get 1280 features
        )
        # Freeze the model weights
        base_model.trainable = False
        self.model = base_model
        print(f"Model loaded. Output shape: {self.model.output_shape}")
        
    def preprocess_mri_slice(self, img_path):

        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return None
                
            # Resize to target size
            img = cv2.resize(img, self.target_size)
            
            # Convert grayscale to RGB (duplicate channels)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Normalize to 0-255 range
            img_rgb = img_rgb.astype(np.float32)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) * 255
            
            # Expand dimensions for batch processing
            img_array = np.expand_dims(img_rgb, axis=0)
            
            # Apply MobileNetV2 preprocessing
            img_processed = preprocess_input(img_array)
            
            return img_processed
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None
    
    def extract_features_from_slice(self, img_path):
        """
        Extract features from a single MRI slice
        
        Args:
            img_path: Path to the MRI slice
            
        Returns:
            Feature vector of shape (1280,)
        """
        preprocessed_img = self.preprocess_mri_slice(img_path)
        if preprocessed_img is None:
            return None
            
        # Extract features using MobileNetV2
        features = self.model.predict(preprocessed_img, verbose=0)
        return features.flatten()
    
    def extract_features_from_case(self, case_folder):
        """
        Extract features from all slices in a case folder and aggregate them
        
        Args:
            case_folder: Path to folder containing MRI slices for one case
            
        Returns:
            Aggregated feature vector for the case
        """
        # Get all image files in the folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.dcm']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(case_folder, ext)))
            image_files.extend(glob.glob(os.path.join(case_folder, ext.upper())))
        
        if not image_files:
            print(f"Warning: No image files found in {case_folder}")
            return None, 0
        
        case_features = []
        processed_slices = 0
        
        print(f"Processing {len(image_files)} slices in {os.path.basename(case_folder)}")
        
        for img_path in tqdm(image_files, desc=f"Extracting features"):
            features = self.extract_features_from_slice(img_path)
            if features is not None:
                case_features.append(features)
                processed_slices += 1
        
        if not case_features:
            print(f"Warning: No valid features extracted from {case_folder}")
            return None, 0
        
        # Aggregate features (mean across all slices)
        case_features = np.array(case_features)
        aggregated_features = np.mean(case_features, axis=0)
        
        return aggregated_features, processed_slices
    
    def process_all_cases(self, data_directory):
        """
        Process all cases in the data directory
        
        Args:
            data_directory: Root directory containing case folders
            
        Returns:
            features_matrix: Array of shape (n_cases, 1280)
            case_names: List of case names
        """
        # Get all subdirectories (case folders)
        case_folders = [d for d in os.listdir(data_directory) 
                       if os.path.isdir(os.path.join(data_directory, d))]
        
        if not case_folders:
            raise ValueError(f"No case folders found in {data_directory}")
        
        print(f"Found {len(case_folders)} cases to process")
        
        all_features = []
        case_names = []
        
        for case_folder in tqdm(case_folders, desc="Processing cases"):
            case_path = os.path.join(data_directory, case_folder)
            features, n_slices = self.extract_features_from_case(case_path)
            
            if features is not None:
                all_features.append(features)
                case_names.append(case_folder)
                print(f"✓ {case_folder}: {n_slices} slices processed")
            else:
                print(f"✗ {case_folder}: Failed to extract features")
        
        if not all_features:
            raise ValueError("No valid features extracted from any case")
        
        features_matrix = np.array(all_features)
        print(f"Extracted features shape: {features_matrix.shape}")
        
        return features_matrix, case_names
    
    def apply_pca(self, features_matrix):
        """
        Apply PCA for dimension reduction
        
        Args:
            features_matrix: Array of shape (n_cases, original_features)
            
        Returns:
            reduced_features: Array of shape (n_cases, n_components)
        """
        print(f"Applying PCA to reduce from {features_matrix.shape[1]} to {self.n_components} dimensions...")
        
        # Standardize features before PCA
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_matrix)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        reduced_features = self.pca.fit_transform(features_scaled)
        
        # Print explained variance
        explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA completed. Explained variance ratio: {explained_variance_ratio:.4f}")
        
        return reduced_features
    
    def save_features_to_csv(self, features, case_names, output_path):
        """
        Save features to CSV file
        
        Args:
            features: Feature matrix
            case_names: List of case names
            output_path: Path to save CSV file
        """
        # Create column names
        feature_columns = [f'feature_{i+1}' for i in range(features.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_columns)
        df.insert(0, 'case_name', case_names)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        print(f"CSV shape: {df.shape}")

def main():
    """Main function to run the feature extraction pipeline"""
    
    # Configuration
    DATA_DIRECTORY = "E:/__Deep Learning/___Her2_ER_PR Paper/mri/data/"  # Update this path
    OUTPUT_CSV = "E:/__Deep Learning/___Her2_ER_PR Paper/FS_1280.csv"     # Output CSV file name
    
    # Initialize the feature extractor
    extractor = MRIFeatureExtractor(target_size=(224, 224), n_components=1280)
    
    try:
        # Load MobileNetV2 model
        extractor.load_mobilenetv2()
        
        # Process all cases and extract features
        print(f"\nStarting feature extraction from: {DATA_DIRECTORY}")
        features_matrix, case_names = extractor.process_all_cases(DATA_DIRECTORY)
        
        # Apply PCA for dimension reduction
        print(f"\nApplying PCA dimension reduction...")
        reduced_features = extractor.apply_pca(features_matrix)
        
        # Save results to CSV
        print(f"\nSaving results...")
        extractor.save_features_to_csv(reduced_features, case_names, OUTPUT_CSV)
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"✓ Processed {len(case_names)} cases")
        print(f"✓ Final feature dimensions: {reduced_features.shape}")
        print(f"✓ Results saved to: {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    print("MRI Feature Extraction Pipeline")
    print("=" * 50)
    
    # Update the data directory path before running
    success = main()
    
    if success:
        print("\nPipeline completed successfully!")
    else:
        print("\nPipeline failed. Please check the error messages above.")

# Additional utility functions
def load_and_preview_results(csv_path):
    """
    Load and preview the results CSV file
    
    Args:
        csv_path: Path to the CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        print(f"First few rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

def visualize_pca_variance(extractor):
    """
    Visualize PCA explained variance (requires matplotlib)
    
    Args:
        extractor: Fitted MRIFeatureExtractor object
    """
    try:
        import matplotlib.pyplot as plt
        
        if extractor.pca is None:
            print("PCA not fitted yet")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(extractor.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error in visualization: {str(e)}")