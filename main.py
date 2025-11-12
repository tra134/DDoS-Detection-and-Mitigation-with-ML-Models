#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup project environment"""
    # Create necessary directories
    directories = [
        'models',
        'data/raw', 'data/processed', 'data/datasets',
        'results', 'logs', 'config'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Project environment setup completed")

def run_ns3_simulation(config_path):
    """Run NS3 simulation"""
    logger.info("Starting NS3 simulation...")
    
    # Load simulation configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Execute NS3 simulation script
    script_path = "scripts/run_ns3_simulation.sh"
    if os.path.exists(script_path):
        os.system(f"bash {script_path}")
    else:
        logger.error("NS3 simulation script not found")
    
    logger.info("NS3 simulation completed")

def train_ml_model(config_path):
    """Train ML model for DDoS detection"""
    logger.info("Training ML model...")
    
    # Import ML pipeline
    from ml_pipeline.train_model import DDoSTrainer
    
    # Load ML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and train model
    trainer = DDoSTrainer(config)
    
    # Load data from NS3 simulation results
    data_path = "data/raw/ns3_detailed_results.csv"
    if os.path.exists(data_path):
        trainer.load_data(data_path)
        results, best_model = trainer.train_models()
        trainer.save_model(best_model, config['output']['model_path'])
    else:
        logger.warning("No simulation data found. Using sample data for training.")
        # Use sample data or synthetic data
        trainer.train_with_sample_data()
    
    logger.info("ML model training completed")

def start_realtime_detection(config_path):
    """Start real-time DDoS detection"""
    logger.info("Starting real-time DDoS detection...")
    
    from ml_pipeline.ddos_detector import RealTimeDDoSDetector
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = RealTimeDDoSDetector(
        model_path=config['output']['model_path'],
        threshold=config['detection']['realtime']['confidence_threshold']
    )
    
    # Start monitoring
    detector.start_monitoring()
    
    logger.info("Real-time detection started")

def analyze_results():
    """Analyze simulation and detection results"""
    logger.info("Analyzing results...")
    
    from analysis.plot_results import ResultAnalyzer
    
    analyzer = ResultAnalyzer()
    analyzer.generate_comprehensive_report()
    
    logger.info("Results analysis completed")

def main():
    parser = argparse.ArgumentParser(description='DDoS Detection and Mitigation System')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project environment')
    
    # NS3 simulation command
    ns3_parser = subparsers.add_parser('simulate', help='Run NS3 simulation')
    ns3_parser.add_argument('--config', default='config/ns3-config.yaml', help='Simulation config file')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--config', default='config/ml-config.yaml', help='ML config file')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Start real-time detection')
    detect_parser.add_argument('--config', default='config/ml-config.yaml', help='Detection config file')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    
    # Complete pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    # Setup environment first
    setup_environment()
    
    if args.command == 'setup':
        logger.info("Environment setup completed")
        
    elif args.command == 'simulate':
        run_ns3_simulation(args.config)
        
    elif args.command == 'train':
        train_ml_model(args.config)
        
    elif args.command == 'detect':
        start_realtime_detection(args.config)
        
    elif args.command == 'analyze':
        analyze_results()
        
    elif args.command == 'pipeline':
        logger.info("Running complete pipeline...")
        run_ns3_simulation('config/ns3-config.yaml')
        train_ml_model('config/ml-config.yaml')
        analyze_results()
        logger.info("Complete pipeline finished")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()