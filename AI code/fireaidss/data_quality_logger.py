

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

class DataQualityLogger:

    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quality tracking
        self.preprocessing_results = {}
        self.sampling_results = {}
        self.scenario_quality = {}
        
    def log_ansys_preprocessing_result(self, scenario: str, timestep: int, 
                                     validation_metrics: Dict, processing_time: float):

        key = f"{scenario}_t{timestep}"
        
        self.preprocessing_results[key] = {
            'scenario': scenario,
            'timestep': timestep,
            'processing_time_seconds': processing_time,
            'temperature_correlation': validation_metrics.get('temperature_correlation', 0.0),
            'spatial_hotspot_preservation': validation_metrics.get('spatial_hotspot_preservation', 0.0),
            'temperature_intensity_preservation': validation_metrics.get('temperature_intensity_preservation', 0.0),
            'fire_source_count_original': validation_metrics.get('fire_source_count_original', 0),
            'fire_source_count_regular': validation_metrics.get('fire_source_count_regular', 0),
            'overall_quality': validation_metrics.get('overall_quality', 0.0),
            'timestamp': time.time()
        }
        
        # Update scenario-level quality tracking
        if scenario not in self.scenario_quality:
            self.scenario_quality[scenario] = []
        self.scenario_quality[scenario].append(validation_metrics.get('overall_quality', 0.0))
        
        # Save incremental results
        self.save_incremental_results()
    
    def log_flight_path_sampling(self, scenario: str, timestep: int, 
                                flight_pattern: str, n_measurements: int,
                                sparse_measurements: List[Dict]):

        key = f"{scenario}_t{timestep}_{flight_pattern}_{n_measurements}"
        
        # Analyze spatial coverage
        coords = np.array([m['coordinates'] for m in sparse_measurements])
        
        # Spatial distribution metrics
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        z_range = coords[:, 2].max() - coords[:, 2].min()
        
        # Measurement spacing analysis
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(coords)
        
        self.sampling_results[key] = {
            'scenario': scenario,
            'timestep': timestep,
            'flight_pattern': flight_pattern,
            'n_measurements': n_measurements,
            'spatial_coverage': {
                'x_range': float(x_range),
                'y_range': float(y_range),
                'z_range': float(z_range),
                'total_volume_coverage': float(x_range * y_range * z_range)
            },
            'measurement_spacing': {
                'min_distance': float(pairwise_distances.min()),
                'max_distance': float(pairwise_distances.max()),
                'mean_distance': float(pairwise_distances.mean()),
                'std_distance': float(pairwise_distances.std())
            },
            'temperature_stats': {
                'min_temp': float(min(m['temperature'] for m in sparse_measurements)),
                'max_temp': float(max(m['temperature'] for m in sparse_measurements)),
                'mean_temp': float(np.mean([m['temperature'] for m in sparse_measurements])),
                'temp_range': float(max(m['temperature'] for m in sparse_measurements) - 
                                  min(m['temperature'] for m in sparse_measurements))
            },
            'timestamp': time.time()
        }
    
    def generate_comprehensive_quality_report(self) -> Dict:

        report = {
            'report_generation_time': time.time(),
            'preprocessing_summary': self.summarize_preprocessing_quality(),
            'sampling_summary': self.summarize_sampling_quality(),
            'scenario_analysis': self.analyze_scenario_quality(),
            'recommendations': self.generate_quality_recommendations()
        }
        
        return report
    
    def summarize_preprocessing_quality(self) -> Dict:

        if not self.preprocessing_results:
            return {'status': 'no_preprocessing_data'}
        
        # Extract metrics
        temp_correlations = [r['temperature_correlation'] for r in self.preprocessing_results.values()]
        hotspot_preservations = [r['spatial_hotspot_preservation'] for r in self.preprocessing_results.values()]
        intensity_preservations = [r['temperature_intensity_preservation'] for r in self.preprocessing_results.values()]
        overall_qualities = [r['overall_quality'] for r in self.preprocessing_results.values()]
        
        return {
            'total_preprocessed_files': len(self.preprocessing_results),
            'temperature_correlation': {
                'mean': np.mean(temp_correlations),
                'std': np.std(temp_correlations),
                'min': np.min(temp_correlations),
                'max': np.max(temp_correlations)
            },
            'spatial_hotspot_preservation': {
                'mean': np.mean(hotspot_preservations),
                'std': np.std(hotspot_preservations),
                'min': np.min(hotspot_preservations),
                'max': np.max(hotspot_preservations),
                'good_quality_count': sum(1 for h in hotspot_preservations if h > 0.8)
            },
            'temperature_intensity_preservation': {
                'mean': np.mean(intensity_preservations),
                'std': np.std(intensity_preservations),
                'min': np.min(intensity_preservations),
                'max': np.max(intensity_preservations)
            },
            'overall_quality': {
                'mean': np.mean(overall_qualities),
                'std': np.std(overall_qualities),
                'good_quality_count': sum(1 for q in overall_qualities if q > 0.7)
            }
        }
    
    def summarize_sampling_quality(self) -> Dict:

        if not self.sampling_results:
            return {'status': 'no_sampling_data'}
        
        # Extract sampling metrics
        measurement_counts = [r['n_measurements'] for r in self.sampling_results.values()]
        spatial_coverages = [r['spatial_coverage']['total_volume_coverage'] for r in self.sampling_results.values()]
        temp_ranges = [r['temperature_stats']['temp_range'] for r in self.sampling_results.values()]
        
        # Flight pattern analysis
        pattern_counts = {}
        for result in self.sampling_results.values():
            pattern = result['flight_pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            'total_sampling_instances': len(self.sampling_results),
            'measurement_count_distribution': {
                'mean': np.mean(measurement_counts),
                'std': np.std(measurement_counts),
                'min': np.min(measurement_counts),
                'max': np.max(measurement_counts)
            },
            'spatial_coverage_analysis': {
                'mean_coverage': np.mean(spatial_coverages),
                'std_coverage': np.std(spatial_coverages),
                'min_coverage': np.min(spatial_coverages),
                'max_coverage': np.max(spatial_coverages)
            },
            'temperature_range_analysis': {
                'mean_range': np.mean(temp_ranges),
                'std_range': np.std(temp_ranges),
                'min_range': np.min(temp_ranges),
                'max_range': np.max(temp_ranges)
            },
            'flight_pattern_distribution': pattern_counts
        }
    
    def analyze_scenario_quality(self) -> Dict:

        scenario_analysis = {}
        
        for scenario, qualities in self.scenario_quality.items():
            scenario_analysis[scenario] = {
                'mean_quality': np.mean(qualities),
                'std_quality': np.std(qualities),
                'min_quality': np.min(qualities),
                'max_quality': np.max(qualities),
                'good_quality_timesteps': sum(1 for q in qualities if q > 0.7),
                'total_timesteps': len(qualities)
            }
        
        return scenario_analysis
    
    def generate_quality_recommendations(self) -> List[str]:

        recommendations = []
        
        # Analyze preprocessing quality
        preprocessing_summary = self.summarize_preprocessing_quality()
        if preprocessing_summary.get('status') != 'no_preprocessing_data':
            avg_hotspot = preprocessing_summary['spatial_hotspot_preservation']['mean']
            avg_correlation = preprocessing_summary['temperature_correlation']['mean']
            
            if avg_hotspot < 0.6:
                recommendations.append("❌ Low hotspot preservation - consider finer grid resolution or different interpolation")
            elif avg_hotspot < 0.8:
                recommendations.append("⚠️  Moderate hotspot preservation - monitor fire source detection performance")
            else:
                recommendations.append("✅ Good hotspot preservation - fire sources well captured")
            
            if avg_correlation < 0.85:
                recommendations.append("❌ Low temperature correlation - check data loading and interpolation")
            elif avg_correlation < 0.95:
                recommendations.append("⚠️  Moderate temperature correlation - acceptable for training")
            else:
                recommendations.append("✅ Excellent temperature correlation - high data fidelity")
        
        # Analyze sampling diversity
        sampling_summary = self.summarize_sampling_quality()
        if sampling_summary.get('status') != 'no_sampling_data':
            pattern_count = len(sampling_summary['flight_pattern_distribution'])
            
            if pattern_count < 3:
                recommendations.append("⚠️  Limited flight pattern diversity - consider more patterns")
            else:
                recommendations.append("✅ Good flight pattern diversity for robust training")
        
        return recommendations
    
    def save_incremental_results(self):

        # Save preprocessing results
        with open(self.output_dir / 'preprocessing_results.json', 'w') as f:
            json.dump(self.preprocessing_results, f, indent=2)
        
        # Save sampling results  
        with open(self.output_dir / 'sampling_results.json', 'w') as f:
            json.dump(self.sampling_results, f, indent=2)
    
    def save_final_quality_report(self):

        report = self.generate_comprehensive_quality_report()
        
        with open(self.output_dir / 'final_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as human-readable text
        with open(self.output_dir / 'quality_summary.txt', 'w') as f:
            f.write("FireAIDSS Data Sampling Quality Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Preprocessing summary
            f.write("ANSYS Preprocessing Quality:\n")
            f.write("-" * 30 + "\n")
            preprocessing = report['preprocessing_summary']
            if preprocessing.get('status') != 'no_preprocessing_data':
                f.write(f"Files processed: {preprocessing['total_preprocessed_files']}\n")
                f.write(f"Avg temperature correlation: {preprocessing['temperature_correlation']['mean']:.3f}\n")
                f.write(f"Avg hotspot preservation: {preprocessing['spatial_hotspot_preservation']['mean']:.3f}\n")
                f.write(f"Good quality files: {preprocessing['overall_quality']['good_quality_count']}/{preprocessing['total_preprocessed_files']}\n\n")
            
            # Sampling summary
            f.write("Flight Path Sampling Quality:\n")
            f.write("-" * 30 + "\n")
            sampling = report['sampling_summary']
            if sampling.get('status') != 'no_sampling_data':
                f.write(f"Sampling instances: {sampling['total_sampling_instances']}\n")
                f.write(f"Measurement count range: {sampling['measurement_count_distribution']['min']}-{sampling['measurement_count_distribution']['max']}\n")
                f.write(f"Flight patterns: {list(sampling['flight_pattern_distribution'].keys())}\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")
            for rec in report['recommendations']:
                f.write(f"{rec}\n")
        
        print(f"📊 Final quality report saved to: {self.output_dir}")
        return report

# Global instance for easy access
_global_quality_logger: Optional[DataQualityLogger] = None

def get_quality_logger(output_dir: Optional[Path] = None) -> DataQualityLogger:

    global _global_quality_logger
    
    if _global_quality_logger is None:
        if output_dir is None:
            output_dir = Path("data_quality_logs")
        _global_quality_logger = DataQualityLogger(output_dir)
    
    return _global_quality_logger

def log_preprocessing_quality(scenario: str, timestep: int, validation_metrics: Dict, processing_time: float):

    logger = get_quality_logger()
    logger.log_ansys_preprocessing_result(scenario, timestep, validation_metrics, processing_time)

def log_sampling_quality(scenario: str, timestep: int, flight_pattern: str, 
                        n_measurements: int, sparse_measurements: List[Dict]):

    logger = get_quality_logger()
    logger.log_flight_path_sampling(scenario, timestep, flight_pattern, n_measurements, sparse_measurements)

def generate_final_quality_report(output_dir: Optional[Path] = None) -> Dict:

    logger = get_quality_logger(output_dir)
    return logger.save_final_quality_report()
