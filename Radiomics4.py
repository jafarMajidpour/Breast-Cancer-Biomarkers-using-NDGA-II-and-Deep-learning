import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import glob
from pathlib import Path
from tqdm import tqdm
import warnings
import logging

# Configure radiomics logging to reduce verbosity
logging.getLogger('radiomics').setLevel(logging.ERROR)

class MRIRadiomicsExtractor:
    def __init__(self, 
                 image_pattern="*image*", 
                 mask_pattern="*mask*",
                 settings=None):
        """
        Initialize the MRI Radiomics Feature Extractor
        
        Args:
            image_pattern: Pattern to match image files (e.g., "*image*", "*img*")
            mask_pattern: Pattern to match mask files (e.g., "*mask*", "*seg*")
            settings: Dictionary of PyRadiomics settings
        """
        self.image_pattern = image_pattern
        self.mask_pattern = mask_pattern
        self.extractor = None
        self.feature_names = None
        
        # Default radiomics settings
        if settings is None:
            self.settings = {
                # Image preprocessing
                'binWidth': 25,  # Bin width for discretization
                'resampledPixelSpacing': None,  # Keep original spacing
                'interpolator': sitk.sitkBSpline,
                'normalizeScale': 100,  # Normalize intensities
                
                # Feature classes to extract
                'enabledImagetypes': {
                    'Original': {},
                    'Wavelet': {},
                    'LoG': {'sigma': [2.0, 3.0, 4.0, 5.0]}
                },
                
                # Distance for texture features
                'distances': [1, 2, 3],
                'force2D': False,  # Set to True for 2D analysis
                'force2Ddimension': 0  # 0 for axial slices
            }
        else:
            self.settings = settings
            
        self.setup_extractor()
    
    def setup_extractor(self):
        """Setup PyRadiomics feature extractor with specified settings"""
        print("Setting up PyRadiomics feature extractor...")
        
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Apply settings
        for key, value in self.settings.items():
            if key == 'enabledImagetypes':
                for img_type, img_settings in value.items():
                    self.extractor.enableImageTypeByName(img_type, **img_settings)
            else:
                self.extractor.addSetting(key, value)
        
        # Enable all feature classes
        self.extractor.enableAllFeatures()
        
        print("Feature extractor configured successfully")
    
    def find_image_mask_pairs(self, case_folder):
        """
        Find image and mask file pairs in a case folder
        
        Args:
            case_folder: Path to case folder
            
        Returns:
            List of (image_path, mask_path) tuples
        """
        # Common medical image extensions
        extensions = ['*.nii', '*.nii.gz', '*.dcm', '*.mha', '*.mhd', 
                     '*.nrrd', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        
        image_files = []
        mask_files = []
        
        # Find image files
        for ext in extensions:
            pattern = os.path.join(case_folder, self.image_pattern + ext)
            image_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(case_folder, self.image_pattern + ext.upper())
            image_files.extend(glob.glob(pattern, recursive=True))
        
        # Find mask files
        for ext in extensions:
            pattern = os.path.join(case_folder, self.mask_pattern + ext)
            mask_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(case_folder, self.mask_pattern + ext.upper())
            mask_files.extend(glob.glob(pattern, recursive=True))
        
        # Sort files for consistent pairing
        image_files.sort()
        mask_files.sort()
        
        # Pair images with masks
        pairs = []
        
        if len(image_files) == 1 and len(mask_files) == 1:
            # Single image-mask pair
            pairs.append((image_files[0], mask_files[0]))
        elif len(image_files) == len(mask_files):
            # Multiple images with corresponding masks
            pairs = list(zip(image_files, mask_files))
        else:
            # Try to match by filename similarity
            for img_file in image_files:
                img_base = Path(img_file).stem.lower()
                best_match = None
                best_score = 0
                
                for mask_file in mask_files:
                    mask_base = Path(mask_file).stem.lower()
                    # Simple similarity based on common substrings
                    common_chars = sum(1 for a, b in zip(img_base, mask_base) if a == b)
                    score = common_chars / max(len(img_base), len(mask_base))
                    
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = mask_file
                
                if best_match:
                    pairs.append((img_file, best_match))
        
        return pairs
    
    def load_medical_image(self, file_path):
        """
        Load medical image using SimpleITK
        
        Args:
            file_path: Path to image file
            
        Returns:
            SimpleITK image object
        """
        try:
            image = sitk.ReadImage(str(file_path))
            return image
        except Exception as e:
            print(f"Error loading image {file_path}: {str(e)}")
            return None
    
    def extract_features_from_pair(self, image_path, mask_path):
        """
        Extract radiomics features from an image-mask pair
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load image and mask
            image = self.load_medical_image(image_path)
            mask = self.load_medical_image(mask_path)
            
            if image is None or mask is None:
                return None
            
            # Ensure same spacing and size
            if image.GetSize() != mask.GetSize():
                print(f"Warning: Size mismatch between image and mask for {image_path}")
                # Resample mask to match image
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                mask = resampler.Execute(mask)
            
            # Extract features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features = self.extractor.execute(image, mask)
            
            # Filter out diagnostic features (keep only actual features)
            filtered_features = {k: v for k, v in features.items() 
                               if not k.startswith('diagnostics_')}
            
            return filtered_features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def process_case(self, case_folder):
        """
        Process all image-mask pairs in a case folder
        
        Args:
            case_folder: Path to case folder
            
        Returns:
            Aggregated features for the case
        """
        case_name = os.path.basename(case_folder)
        
        # Find image-mask pairs
        pairs = self.find_image_mask_pairs(case_folder)
        
        if not pairs:
            print(f"Warning: No valid image-mask pairs found in {case_folder}")
            return None, 0
        
        print(f"Found {len(pairs)} image-mask pairs in {case_name}")
        
        all_features = []
        processed_pairs = 0
        
        for img_path, mask_path in tqdm(pairs, desc=f"Processing {case_name}"):
            features = self.extract_features_from_pair(img_path, mask_path)
            
            if features is not None:
                all_features.append(features)
                processed_pairs += 1
            else:
                print(f"Failed to process pair: {os.path.basename(img_path)}")
        
        if not all_features:
            print(f"Warning: No valid features extracted from {case_folder}")
            return None, 0
        
        # Store feature names from first successful extraction
        if self.feature_names is None:
            self.feature_names = list(all_features[0].keys())
        
        # Convert to DataFrame for easier aggregation
        features_df = pd.DataFrame(all_features)
        
        # Aggregate features across slices (mean)
        aggregated_features = features_df.mean().to_dict()
        
        return aggregated_features, processed_pairs
    
    def process_all_cases(self, data_directory):
        """
        Process all cases in the data directory
        
        Args:
            data_directory: Root directory containing case folders
            
        Returns:
            DataFrame with all extracted features
        """
        # Get all case folders
        case_folders = [d for d in os.listdir(data_directory) 
                       if os.path.isdir(os.path.join(data_directory, d))]
        
        if not case_folders:
            raise ValueError(f"No case folders found in {data_directory}")
        
        print(f"Found {len(case_folders)} cases to process")
        
        all_case_features = []
        case_names = []
        
        for case_folder in tqdm(case_folders, desc="Processing cases"):
            case_path = os.path.join(data_directory, case_folder)
            features, n_pairs = self.process_case(case_path)
            
            if features is not None:
                features['case_name'] = case_folder
                all_case_features.append(features)
                case_names.append(case_folder)
                print(f"✓ {case_folder}: {n_pairs} image-mask pairs processed")
            else:
                print(f"✗ {case_folder}: Failed to extract features")
        
        if not all_case_features:
            raise ValueError("No valid features extracted from any case")
        
        # Create DataFrame
        features_df = pd.DataFrame(all_case_features)
        
        # Reorder columns to have case_name first
        cols = ['case_name'] + [col for col in features_df.columns if col != 'case_name']
        features_df = features_df[cols]
        
        print(f"Extracted features shape: {features_df.shape}")
        print(f"Feature types: {len([col for col in features_df.columns if col != 'case_name'])} radiomics features")
        
        return features_df
    
    def save_features_to_csv(self, features_df, output_path, include_feature_info=True):
        """
        Save features to CSV file with optional feature information
        
        Args:
            features_df: DataFrame containing features
            output_path: Path to save CSV file
            include_feature_info: Whether to save feature information separately
        """
        # Save main features
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        print(f"CSV shape: {features_df.shape}")
        
        if include_feature_info and self.feature_names:
            # Save feature information
            info_path = output_path.replace('.csv', '_feature_info.csv')
            feature_info = []
            
            for feature_name in self.feature_names:
                # Parse feature name to extract class and type
                parts = feature_name.split('_')
                if len(parts) >= 3:
                    image_type = parts[0]
                    feature_class = parts[1]
                    feature_name_only = '_'.join(parts[2:])
                else:
                    image_type = "Unknown"
                    feature_class = "Unknown"
                    feature_name_only = feature_name
                
                feature_info.append({
                    'feature_name': feature_name,
                    'image_type': image_type,
                    'feature_class': feature_class,
                    'feature_name_only': feature_name_only
                })
            
            info_df = pd.DataFrame(feature_info)
            info_df.to_csv(info_path, index=False)
            print(f"Feature information saved to {info_path}")

def create_sample_settings():
    """
    Create sample settings for different radiomics extraction scenarios
    
    Returns:
        Dictionary of different setting configurations
    """
    settings_configs = {
        'basic_2d': {
            'binWidth': 25,
            'force2D': True,
            'force2Ddimension': 0,  # Axial slices
            'enabledImagetypes': {'Original': {}},
            'distances': [1, 2, 3]
        },
        
        'comprehensive_2d': {
            'binWidth': 25,
            'force2D': True,
            'force2Ddimension': 0,
            'enabledImagetypes': {
                'Original': {},
                'Wavelet': {},
                'LoG': {'sigma': [2.0, 3.0, 4.0, 5.0]},
                'Square': {},
                'SquareRoot': {},
                'Logarithm': {},
                'Exponential': {}
            },
            'distances': [1, 2, 3]
        },
        
        'basic_3d': {
            'binWidth': 25,
            'force2D': False,
            'enabledImagetypes': {'Original': {}},
            'distances': [1, 2, 3]
        },
        
        'comprehensive_3d': {
            'binWidth': 25,
            'force2D': False,
            'enabledImagetypes': {
                'Original': {},
                'Wavelet': {},
                'LoG': {'sigma': [2.0, 3.0, 4.0, 5.0]}
            },
            'distances': [1, 2, 3, 4, 5]
        }
    }
    
    return settings_configs

def main():
    """Main function to run the radiomics feature extraction pipeline"""
    
    # Configuration
    DATA_DIRECTORY = "path/to/your/mri/data"  # Update this path
    OUTPUT_CSV = "mri_radiomics_features.csv"
    
    # Choose settings configuration
    settings_configs = create_sample_settings()
    chosen_settings = settings_configs['comprehensive_2d']  # Change as needed
    
    # Initialize the feature extractor
    extractor = MRIRadiomicsExtractor(
        image_pattern="*image*",  # Adjust pattern to match your image files
        mask_pattern="*mask*",    # Adjust pattern to match your mask files
        settings=chosen_settings
    )
    
    try:
        print(f"Starting radiomics feature extraction from: {DATA_DIRECTORY}")
        print(f"Image pattern: {extractor.image_pattern}")
        print(f"Mask pattern: {extractor.mask_pattern}")
        print(f"Settings: {list(chosen_settings.keys())}")
        
        # Process all cases
        features_df = extractor.process_all_cases(DATA_DIRECTORY)
        
        # Save results
        print(f"\nSaving results...")
        extractor.save_features_to_csv(features_df, OUTPUT_CSV, include_feature_info=True)
        
        # Print summary statistics
        print(f"\n" + "="*60)
        print(f"EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"="*60)
        print(f"✓ Processed cases: {len(features_df)}")
        print(f"✓ Total features per case: {len(features_df.columns) - 1}")
        print(f"✓ Results saved to: {OUTPUT_CSV}")
        
        # Feature breakdown by class
        feature_classes = {}
        for col in features_df.columns:
            if col != 'case_name':
                parts = col.split('_')
                if len(parts) >= 2:
                    class_name = parts[1]
                    feature_classes[class_name] = feature_classes.get(class_name, 0) + 1
        
        print(f"\nFeature breakdown by class:")
        for class_name, count in sorted(feature_classes.items()):
            print(f"  {class_name}: {count} features")
        
        return features_df
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return None

def load_and_analyze_results(csv_path):
    """
    Load and analyze the results CSV file
    
    Args:
        csv_path: Path to the results CSV file
    """
    csv_path= 'E:/__Deep Learning/___Her2_ER_PR Paper/Radiomics.csv'
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded results: {df.shape}")
        
        # Basic statistics
        print(f"\nBasic statistics:")
        print(f"  Cases: {len(df)}")
        print(f"  Features: {len(df.columns) - 1}")
        
        # Show first few columns
        print(f"\nFirst 5 feature columns:")
        feature_cols = [col for col in df.columns if col != 'case_name'][:5]
        print(df[['case_name'] + feature_cols].head())
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing values")
        else:
            print(f"\n✓ No missing values found")
        
        return df
        
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def create_feature_extractor_with_custom_features():
    """
    Example of creating extractor with specific feature classes
    
    Returns:
        MRIRadiomicsExtractor with custom settings
    """
    custom_settings = {
        'binWidth': 25,
        'force2D': True,
        'force2Ddimension': 0,
        'enabledImagetypes': {'Original': {}},
        'distances': [1, 2, 3]
    }
    
    extractor = MRIRadiomicsExtractor(settings=custom_settings)
    
    # Disable all features first
    extractor.extractor.disableAllFeatures()
    
    # Enable specific feature classes
    extractor.extractor.enableFeatureClassByName('firstorder')
    extractor.extractor.enableFeatureClassByName('shape')
    extractor.extractor.enableFeatureClassByName('glcm')
    extractor.extractor.enableFeatureClassByName('glrlm')
    extractor.extractor.enableFeatureClassByName('glszm')
    extractor.extractor.enableFeatureClassByName('gldm')
    extractor.extractor.enableFeatureClassByName('ngtdm')
    
    return extractor

# Utility function for DICOM series handling
def process_dicom_series(case_folder):
    """
    Handle DICOM series by converting to NIfTI format
    
    Args:
        case_folder: Path to folder containing DICOM files
        
    Returns:
        Path to converted NIfTI file
    """
    case_folder= 'E:/__Deep Learning/___Her2_ER_PR Paper/mri/data/'
    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(case_folder)
        
        if not dicom_names:
            return None
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Save as NIfTI
        output_path = os.path.join(case_folder, 'converted_image.nii.gz')
        sitk.WriteImage(image, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error processing DICOM series in {case_folder}: {str(e)}")
        return None

if __name__ == "__main__":
    print("MRI Radiomics Feature Extraction Pipeline")
    print("=" * 60)
    
    # Example usage with different configurations
    print("\nAvailable settings configurations:")
    configs = create_sample_settings()
    for name, config in configs.items():
        print(f"  {name}: {config.get('force2D', 'Unknown')} analysis")
    
    print(f"\nTo run the pipeline:")
    print(f"1. Update DATA_DIRECTORY path")
    print(f"2. Adjust image_pattern and mask_pattern to match your files")
    print(f"3. Choose appropriate settings configuration")
    print(f"4. Run main()")
    
    # Uncomment to run
    # results = main()
    
    # Example of loading and analyzing results
    # df = load_and_analyze_results("mri_radiomics_features.csv")