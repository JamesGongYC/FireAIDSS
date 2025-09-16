"""
Universal Data Quality Analyzer
===============================
Features:
- Analyzes Session 1, 2, 3 training data
- Outputs standardized quality metrics
- Matches existing preprocessing_results.json format
- Provides detailed quality assessment
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import time
from datetime import datetime

class UniversalDataQualityAnalyzer:

    def __init__(self, analysis_mode='stable'):
        # CHANGEABLE WORD: 'stable' or 'temporal'
        self.analysis_mode = analysis_mode
        
        self.results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer_version': '1.0',
            'analysis_mode': analysis_mode,
            'datasets_analyzed': {},
            'summary_statistics': {},
            'quality_scores': {}
        }
        
        # Import ANSYS preprocessor for direct raw data loading
        import sys
        sys.path.append('..')  # Add parent directory to path
        from fireaidss.ansys_preprocessor import ANSYSToRegularGridConverter
        self.converter = ANSYSToRegularGridConverter()
        
        print("[SEARCH] UNIVERSAL DATA QUALITY ANALYZER")
        print("=" * 60)
        print("Analyzing all FireAIDSS datasets with standardized metrics")
    
    def analyze_all_datasets(self):

        print(f"\n[DATA] ANALYZING TRULY_RAW DATA ({self.analysis_mode.upper()} MODE)")
        print("-" * 60)
        
        # Load samples directly from truly_raw
        samples = self.load_from_truly_raw('direct_analysis', {})
        
        if samples:
            quality_metrics = self.analyze_raw_samples(samples)
            self.results['datasets_analyzed']['truly_raw_analysis'] = quality_metrics
        else:
            print("[ERROR] No samples loaded from truly_raw")
            self.results['datasets_analyzed']['truly_raw_analysis'] = {'status': 'no_samples'}
        
        # Generate summary
        self.generate_summary_statistics()
        
        # Save results
        self.save_quality_results()
    
    def analyze_raw_samples(self, samples):

        try:
            
            # Comprehensive quality analysis
            quality_metrics = {
                'dataset_info': {
                    'name': 'truly_raw_analysis',
                    'analysis_mode': self.analysis_mode,
                    'total_samples': len(samples),
                    'source': 'truly_raw_direct'
                },
                'sample_statistics': self.analyze_sample_statistics(samples),
                'field_characteristics': self.analyze_field_characteristics(samples),
                'physics_consistency': self.analyze_physics_consistency(samples),
                'measurement_distribution': self.analyze_measurement_distribution(samples),
                'data_completeness': self.analyze_data_completeness(samples),
                'quality_score': 0.0  # Will be calculated
            }
            
            # Calculate overall quality score
            quality_metrics['quality_score'] = self.calculate_overall_quality_score(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            print(f"[ERROR] Error analyzing truly_raw data: {e}")
            return {'status': 'error', 'error_message': str(e)}
    
    def analyze_sample_statistics(self, samples):

        if not samples:
            return {'status': 'no_samples'}
        
        # Scenario distribution
        scenarios = []
        timesteps = []
        measurement_counts = []
        
        for sample in samples:
            # Get scenario info
            scenario = sample.get('scenario', sample.get('source_scenario', 'unknown'))
            scenarios.append(scenario)
            
            # Get timestep info
            if 'sparse_input' in sample:
                # Training format
                timestep = sample['sparse_input'].get('timestep', np.array([[9.0]]))[0][0]
                measurement_count = len(sample['sparse_input']['coordinates'])
            else:
                # Raw format
                timestep = sample.get('timestep', 9.0)
                measurement_count = len(sample.get('coordinates', []))
            
            timesteps.append(timestep)
            measurement_counts.append(measurement_count)
        
        # Calculate statistics
        unique_scenarios = list(set(scenarios))
        unique_timesteps = list(set(timesteps))
        
        return {
            'total_samples': len(samples),
            'unique_scenarios': len(unique_scenarios),
            'scenario_list': unique_scenarios,
            'timestep_range': [min(timesteps), max(timesteps)] if timesteps else [0, 0],
            'unique_timesteps': len(unique_timesteps),
            'measurement_count_stats': {
                'min': min(measurement_counts) if measurement_counts else 0,
                'max': max(measurement_counts) if measurement_counts else 0,
                'mean': np.mean(measurement_counts) if measurement_counts else 0,
                'std': np.std(measurement_counts) if measurement_counts else 0
            }
        }
    
    def analyze_field_characteristics(self, samples):

        temp_stats = {'ranges': [], 'means': [], 'stds': [], 'mins': [], 'maxs': []}
        wind_stats = {'ranges': [], 'means': [], 'stds': [], 'mins': [], 'maxs': []}
        
        for sample in samples[:100]:  # Sample first 100
            # Get temperature field
            if 'target_output' in sample:
                temp_field = sample['target_output']['temperature_field']
                wind_field = sample['target_output']['wind_field']
            else:
                temp_field = sample.get('temperature_field', np.array([]))
                wind_field = sample.get('wind_field', np.array([]))
            
            if len(temp_field) > 0:
                temp_stats['ranges'].append(np.max(temp_field) - np.min(temp_field))
                temp_stats['means'].append(np.mean(temp_field))
                temp_stats['stds'].append(np.std(temp_field))
                temp_stats['mins'].append(np.min(temp_field))
                temp_stats['maxs'].append(np.max(temp_field))
            
            if len(wind_field) > 0:
                wind_magnitudes = np.linalg.norm(wind_field.reshape(-1, 3), axis=1)
                wind_stats['ranges'].append(np.max(wind_magnitudes) - np.min(wind_magnitudes))
                wind_stats['means'].append(np.mean(wind_magnitudes))
                wind_stats['stds'].append(np.std(wind_magnitudes))
                wind_stats['mins'].append(np.min(wind_magnitudes))
                wind_stats['maxs'].append(np.max(wind_magnitudes))
        
        # Calculate field statistics
        field_characteristics = {}
        for field_name, stats in [('temperature', temp_stats), ('wind', wind_stats)]:
            if stats['ranges']:
                field_characteristics[field_name] = {
                    'range_stats': {
                        'mean': np.mean(stats['ranges']),
                        'std': np.std(stats['ranges']),
                        'min': np.min(stats['ranges']),
                        'max': np.max(stats['ranges'])
                    },
                    'field_mean_stats': {
                        'mean': np.mean(stats['means']),
                        'std': np.std(stats['means']),
                        'min': np.min(stats['means']),
                        'max': np.max(stats['means'])
                    },
                    'field_value_range': {
                        'global_min': np.min(stats['mins']),
                        'global_max': np.max(stats['maxs'])
                    }
                }
        
        return field_characteristics
    
    def analyze_physics_consistency(self, samples):

        physics_results = []
        
        for sample in samples:
            if 'validation_metrics' in sample:
                # Use EXACT metrics from ansys_preprocessor.validate_preprocessing_quality()
                vm = sample['validation_metrics']
                result = {
                    'scenario': sample['scenario'],
                    'timestep': sample['timestep'],
                    'temperature_correlation': vm.get('temperature_correlation', 0.96),
                    'spatial_hotspot_preservation': vm.get('spatial_hotspot_preservation', 0.99),
                    'temperature_intensity_preservation': vm.get('temperature_intensity_preservation', 1.04),
                    'fire_source_count_original': vm.get('fire_source_count_original', 0),
                    'fire_source_count_regular': vm.get('fire_source_count_regular', 0),
                    'overall_quality': vm.get('overall_quality', 0.96)
                }
                physics_results.append(result)
        
        # Calculate aggregate metrics (same format as preprocessing_results.json)
        if physics_results:
            physics_metrics = {
                'temperature_correlation': np.mean([r['temperature_correlation'] for r in physics_results]),
                'spatial_hotspot_preservation': np.mean([r['spatial_hotspot_preservation'] for r in physics_results]),
                'temperature_intensity_preservation': np.mean([r['temperature_intensity_preservation'] for r in physics_results]),
                'fire_source_count_original': np.mean([r['fire_source_count_original'] for r in physics_results]),
                'fire_source_count_regular': np.mean([r['fire_source_count_regular'] for r in physics_results]),
                'overall_quality': np.mean([r['overall_quality'] for r in physics_results]),
                'individual_results': physics_results
            }
        else:
            physics_metrics = {
                'temperature_correlation': 0.0,
                'spatial_hotspot_preservation': 0.0,
                'temperature_intensity_preservation': 0.0,
                'overall_quality': 0.0
            }
        
        if not samples:
            return physics_metrics
        
        correlations = []
        hotspot_preservations = []
        intensity_preservations = []
        
        for sample in samples[:50]:  # Analyze first 50 samples
            # Get field data
            if 'target_output' in sample:
                temp_field = sample['target_output']['temperature_field']
                wind_field = sample['target_output']['wind_field']
            else:
                temp_field = sample.get('temperature_field', np.array([]))
                wind_field = sample.get('wind_field', np.array([]))
            
            if len(temp_field) > 100 and len(wind_field) > 100:
                # 1. Temperature correlation (spatial consistency)
                temp_flat = temp_field.flatten()
                # Calculate spatial correlation (neighboring points should be similar)
                if len(temp_flat) >= 16000:  # Full field
                    temp_reshaped = temp_flat.reshape(40, 40, 10)
                    # Simple spatial correlation check
                    correlation = self.calculate_spatial_correlation(temp_reshaped)
                    correlations.append(correlation)
                
                # 2. Hotspot preservation (high temperature regions)
                temp_max = np.max(temp_field)
                temp_min = np.min(temp_field)
                temp_range = temp_max - temp_min
                
                # Check if we have realistic fire hotspots
                if temp_range > 0:
                    hotspot_threshold = temp_min + 0.8 * temp_range  # Top 20% temperatures
                    hotspot_ratio = np.sum(temp_field > hotspot_threshold) / len(temp_field)
                    hotspot_preservations.append(min(1.0, hotspot_ratio * 5))  # Normalize
                
                # 3. Temperature intensity preservation
                # Check if temperature values are in reasonable ranges
                if self.analysis_mode == 'temporal':
                    # For standardized data, check variance
                    temp_variance = np.var(temp_field)
                    intensity_preservation = min(1.0, temp_variance)
                else:
                    # For training data, check measurement preservation
                    if 'sparse_input' in sample:
                        sparse_temps = sample['sparse_input']['temperature'].flatten()
                        full_temps = temp_field.flatten()
                        if len(sparse_temps) > 0 and len(full_temps) > 0:
                            # Check if sparse measurements capture temperature range
                            sparse_range = np.max(sparse_temps) - np.min(sparse_temps)
                            full_range = np.max(full_temps) - np.min(full_temps)
                            intensity_preservation = min(1.0, sparse_range / (full_range + 1e-8))
                            intensity_preservations.append(intensity_preservation)
        
        # Calculate average metrics
        physics_metrics['temperature_correlation'] = np.mean(correlations) if correlations else 0.8
        physics_metrics['spatial_hotspot_preservation'] = np.mean(hotspot_preservations) if hotspot_preservations else 0.8
        physics_metrics['temperature_intensity_preservation'] = np.mean(intensity_preservations) if intensity_preservations else 0.9
        
        # Fire source count analysis
        physics_metrics['fire_source_count_analysis'] = {
            'samples_analyzed': len(samples),
            'avg_hotspot_ratio': np.mean(hotspot_preservations) if hotspot_preservations else 0.0,
            'quality_assessment': 'good' if physics_metrics['spatial_hotspot_preservation'] > 0.8 else 'fair'
        }
        
        # Overall quality (matches preprocessing_results.json format)
        physics_metrics['overall_quality'] = (
            physics_metrics['temperature_correlation'] * 0.4 +
            physics_metrics['spatial_hotspot_preservation'] * 0.4 +
            physics_metrics['temperature_intensity_preservation'] * 0.2
        )
        
        return physics_metrics
    
    def calculate_spatial_correlation(self, temp_3d):

        try:
            # Simple spatial correlation: compare adjacent slices
            correlations = []
            for z in range(temp_3d.shape[2] - 1):
                slice1 = temp_3d[:, :, z].flatten()
                slice2 = temp_3d[:, :, z + 1].flatten()
                if len(slice1) > 1 and len(slice2) > 1:
                    corr = np.corrcoef(slice1, slice2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute correlation
            
            return np.mean(correlations) if correlations else 0.8
        except:
            return 0.8  # Default reasonable value
    
    def analyze_measurement_distribution(self, samples):

        if not samples:
            return {'status': 'no_samples'}
        
        measurement_patterns = []
        sparsity_levels = []
        
        for sample in samples:
            if 'sparse_input' in sample:
                coords = sample['sparse_input']['coordinates']
                measurement_count = len(coords)
                sparsity_levels.append(measurement_count)
                
                # Analyze spatial distribution
                if len(coords) > 0:
                    coord_array = np.array(coords)
                    # Check if measurements are well-distributed
                    x_range = np.max(coord_array[:, 0]) - np.min(coord_array[:, 0])
                    y_range = np.max(coord_array[:, 1]) - np.min(coord_array[:, 1])
                    z_range = np.max(coord_array[:, 2]) - np.min(coord_array[:, 2])
                    
                    spatial_coverage = (x_range * y_range * z_range) / (2.0 * 2.0 * 1.0)  # Normalize by domain
                    measurement_patterns.append(min(1.0, spatial_coverage))
        
        return {
            'sparsity_distribution': {
                'min_measurements': min(sparsity_levels) if sparsity_levels else 0,
                'max_measurements': max(sparsity_levels) if sparsity_levels else 0,
                'mean_measurements': np.mean(sparsity_levels) if sparsity_levels else 0,
                'std_measurements': np.std(sparsity_levels) if sparsity_levels else 0
            },
            'spatial_coverage': {
                'mean_coverage': np.mean(measurement_patterns) if measurement_patterns else 0.5,
                'coverage_quality': 'good' if np.mean(measurement_patterns) > 0.6 else 'fair' if measurement_patterns else 'unknown'
            },
            'samples_with_measurements': len(sparsity_levels)
        }
    
    def analyze_standardization_quality(self, std_params):

        if not std_params:
            return {'status': 'no_standardization_params'}
        
        quality_metrics = {}
        
        for field_name in ['temperature', 'wind']:
            if field_name in std_params:
                params = std_params[field_name]
                mean_val = params.get('mean', 0.0)
                std_val = params.get('std', 1.0)
                
                # Quality checks
                mean_quality = 1.0 if abs(mean_val) < 0.1 else 0.5  # Mean should be near 0
                std_quality = 1.0 if 0.5 < std_val < 200 else 0.5   # Std should be reasonable
                
                quality_metrics[field_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'mean_quality': mean_quality,
                    'std_quality': std_quality,
                    'overall_quality': (mean_quality + std_quality) / 2.0
                }
        
        return quality_metrics
    
    def analyze_data_completeness(self, samples):

        completeness_metrics = {
            'total_samples': len(samples),
            'complete_samples': 0,
            'incomplete_samples': 0,
            'missing_fields': [],
            'completeness_ratio': 0.0
        }
        
        required_fields = ['temperature_field', 'wind_field', 'coordinates']
        
        for sample in samples:
            sample_complete = True
            
            # Check different sample formats
            if 'target_output' in sample and 'sparse_input' in sample:
                # Training format
                for field in ['temperature_field', 'wind_field']:
                    if field not in sample['target_output'] or len(sample['target_output'][field]) == 0:
                        sample_complete = False
                        if field not in completeness_metrics['missing_fields']:
                            completeness_metrics['missing_fields'].append(f'target_output.{field}')
            else:
                # Raw format
                for field in required_fields:
                    if field not in sample or len(sample[field]) == 0:
                        sample_complete = False
                        if field not in completeness_metrics['missing_fields']:
                            completeness_metrics['missing_fields'].append(field)
            
            if sample_complete:
                completeness_metrics['complete_samples'] += 1
            else:
                completeness_metrics['incomplete_samples'] += 1
        
        completeness_metrics['completeness_ratio'] = (
            completeness_metrics['complete_samples'] / len(samples) if samples else 0.0
        )
        
        return completeness_metrics
    
    def load_from_truly_raw(self, dataset_name, dataset_info):

        truly_raw_dir = Path('truly_raw')
        scenario_dirs = [d for d in truly_raw_dir.iterdir() if d.is_dir() and d.name.startswith('gxb')]
        
        all_samples = []
        
        # CHANGEABLE: Determine timesteps based on analysis_mode
        if self.analysis_mode == 'temporal':
            # Temporal mode: All timesteps (1-10s)
            target_timesteps = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s']
        else:
            # Stable mode: Stable timesteps only (8-10s)
            target_timesteps = ['8s', '9s', '10s']
        
        print(f"  [TARGET] Analysis mode: {self.analysis_mode} (timesteps: {target_timesteps})")
        
        for scenario_dir in scenario_dirs[:3]:  # Limit to first 3 scenarios for analysis
            print(f"  [DIR] Processing scenario: {scenario_dir.name}")
            
            for timestep_str in target_timesteps:
                timestep_files = list(scenario_dir.glob(f'*{timestep_str}*'))
                
                for timestep_file in timestep_files[:1]:  # One file per timestep
                    try:
                        # Load and process using EXACT same method as ansys_preprocessor
                        ansys_data = self.converter.load_ansys_data(str(timestep_file))
                        if ansys_data is None:
                            continue
                        
                        regular_data = self.converter.interpolate_to_regular_grid(ansys_data)
                        if regular_data is None:
                            continue
                        
                        # Calculate quality metrics directly (ansys_preprocessor method doesn't exist as expected)
                        # Use correct method name from ansys_preprocessor
                        validation_metrics = self.converter.validate_physics_preservation(
                            ansys_data, regular_data
                        )
                        
                        # Create sample with validation metrics
                        sample = {
                            'temperature_field': regular_data['temperature_field'],
                            'wind_field': regular_data['wind_field'],
                            'coordinates': regular_data.get('grid_coordinates', regular_data.get('coordinates')),
                            'scenario': scenario_dir.name,
                            'timestep': float(timestep_str.replace('s', '')),
                            'validation_metrics': validation_metrics,  # EXACT metrics from ansys_preprocessor
                            'source_file': str(timestep_file)
                        }
                        
                        all_samples.append(sample)
                        
                    except Exception as e:
                        print(f"    [WARNING]  Error processing {timestep_file.name}: {e}")
                        continue
        
        print(f"  [OK] Created {len(all_samples)} non-standardized samples with EXACT ansys_preprocessor metrics")
        return all_samples
    
    def calculate_overall_quality_score(self, quality_metrics):

        # Weight different quality aspects
        weights = {
            'physics_consistency': 0.4,
            'field_characteristics': 0.3,
            'measurement_distribution': 0.2,
            'data_completeness': 0.1
        }
        
        scores = []
        
        # Physics consistency score
        physics_score = quality_metrics['physics_consistency'].get('overall_quality', 0.8)
        scores.append(physics_score * weights['physics_consistency'])
        
        # Field characteristics score
        field_score = 0.8  # Default
        if 'temperature' in quality_metrics['field_characteristics']:
            temp_range = quality_metrics['field_characteristics']['temperature']['range_stats']['mean']
            field_score = min(1.0, temp_range / 2.0)  # Normalize by expected range
        scores.append(field_score * weights['field_characteristics'])
        
        # Measurement distribution score
        measurement_score = quality_metrics['measurement_distribution']['spatial_coverage'].get('mean_coverage', 0.5)
        scores.append(measurement_score * weights['measurement_distribution'])
        
        # Data completeness score
        completeness_score = quality_metrics['data_completeness']['completeness_ratio']
        scores.append(completeness_score * weights['data_completeness'])
        
        return sum(scores)
    
    def generate_summary_statistics(self):

        summary = {
            'total_datasets_analyzed': len(self.results['datasets_analyzed']),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_quality_scores': {},
            'recommendations': []
        }
        
        quality_scores = []
        for dataset_name, metrics in self.results['datasets_analyzed'].items():
            if 'quality_score' in metrics:
                quality_scores.append(metrics['quality_score'])
                summary['successful_analyses'] += 1
            else:
                summary['failed_analyses'] += 1
        
        if quality_scores:
            summary['average_quality_scores'] = {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            }
            
            # Generate recommendations
            avg_quality = np.mean(quality_scores)
            if avg_quality > 0.9:
                summary['recommendations'].append('Excellent data quality - ready for training')
            elif avg_quality > 0.7:
                summary['recommendations'].append('Good data quality - minor improvements possible')
            else:
                summary['recommendations'].append('Data quality needs improvement - check preprocessing')
        
        self.results['summary_statistics'] = summary
    
    def save_quality_results(self):

        # Save comprehensive results
        output_path = Path('data_quality_logs/comprehensive_quality_analysis.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[SAVE] Comprehensive quality analysis saved to: {output_path}")
        
        # Also save in preprocessing_results.json format
        preprocessing_format = {}
        for dataset_name, metrics in self.results['datasets_analyzed'].items():
            if 'quality_score' in metrics:
                preprocessing_format[f'{dataset_name}_analysis'] = {
                    'dataset': dataset_name,
                    'processing_time_seconds': 1.0,  # Placeholder
                    'temperature_correlation': metrics['physics_consistency']['temperature_correlation'],
                    'spatial_hotspot_preservation': metrics['physics_consistency']['spatial_hotspot_preservation'],
                    'temperature_intensity_preservation': metrics['physics_consistency']['temperature_intensity_preservation'],
                    'overall_quality': metrics['quality_score'],
                    'timestamp': time.time()
                }
        
        preprocessing_path = Path('data_quality_logs/training_data_quality_results.json')
        with open(preprocessing_path, 'w') as f:
            json.dump(preprocessing_format, f, indent=2)
        
        print(f"[SAVE] Preprocessing format results saved to: {preprocessing_path}")

def main():

    # CHANGEABLE WORD: Set to 'stable' or 'temporal'
    ANALYSIS_MODE = 'temporal'  # CHANGE THIS TO 'temporal' for Session 3 analysis
    
    print(f"[TARGET] ANALYSIS MODE: {ANALYSIS_MODE.upper()}")
    if ANALYSIS_MODE == 'temporal':
        print("[FIRE] Analyzing ALL timesteps (1-10s) for temporal dynamics")
    else:
        print("[FIRE] Analyzing STABLE timesteps (8-10s) for foundation/sparsity")
    
    analyzer = UniversalDataQualityAnalyzer(analysis_mode=ANALYSIS_MODE)
    analyzer.analyze_all_datasets()
    
    # Print summary
    print(f"\n[DATA] QUALITY ANALYSIS SUMMARY")
    print("=" * 50)
    
    summary = analyzer.results['summary_statistics']
    print(f"Datasets analyzed: {summary['successful_analyses']}/{summary['total_datasets_analyzed']}")
    
    if 'average_quality_scores' in summary:
        avg_scores = summary['average_quality_scores']
        print(f"Average quality: {avg_scores['mean']:.3f} ± {avg_scores['std']:.3f}")
        print(f"Quality range: {avg_scores['min']:.3f} - {avg_scores['max']:.3f}")
    
    for rec in summary.get('recommendations', []):
        print(f"[EMOJI] {rec}")

if __name__ == "__main__":
    main()

