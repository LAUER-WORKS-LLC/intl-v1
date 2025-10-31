#!/usr/bin/env python3
"""
Weekly Breakout Prediction Experiment Runner
Main script for running the experiment on EC2 with proper monitoring
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
import argparse
import json
import yaml
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weekly_breakout_experiment import WeeklyBreakoutExperiment
from monitor_experiment import ExperimentMonitor

class ExperimentRunner:
    """Main experiment runner with monitoring and error handling"""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        self.config_path = config_path
        self.experiment = None
        self.monitor = ExperimentMonitor()
        self.start_time = None
        self.run_id = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        if self.experiment:
            self._save_progress()
        sys.exit(0)
    
    def _save_progress(self):
        """Save current progress"""
        try:
            progress_data = {
                'run_id': self.run_id,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'current_time': datetime.now().isoformat(),
                'status': 'interrupted'
            }
            
            with open('results/progress.json', 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving progress: {str(e)}")
    
    def run_experiment(self, background: bool = False):
        """Run the complete experiment"""
        self.start_time = datetime.now()
        self.run_id = f"weekly_breakout_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 Starting Weekly Breakout Prediction Experiment")
        print(f"Run ID: {self.run_id}")
        print(f"Start Time: {self.start_time}")
        print(f"Background Mode: {background}")
        print("="*60)
        
        try:
            # Initialize experiment
            self.experiment = WeeklyBreakoutExperiment(self.config_path)
            self.experiment.run_id = self.run_id
            
            # Save initial progress
            self._save_progress()
            
            # Run experiment
            if background:
                self._run_in_background()
            else:
                self._run_foreground()
            
            print("✅ Experiment completed successfully!")
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            self._save_progress()
            raise
    
    def _run_foreground(self):
        """Run experiment in foreground"""
        if self.experiment is None:
            self.experiment = WeeklyBreakoutExperiment(self.config_path)
        self.experiment.run_experiment()
    
    def _run_in_background(self):
        """Run experiment in background with monitoring"""
        import subprocess
        import nohup
        
        # Start experiment in background
        cmd = [sys.executable, __file__, '--foreground']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"🔄 Experiment started in background (PID: {process.pid})")
        print("Use 'python monitor_experiment.py --check' to check status")
        print("Use 'python monitor_experiment.py --monitor' for continuous monitoring")
    
    def check_status(self):
        """Check experiment status"""
        status = self.monitor.check_experiment_status()
        progress = self.monitor.get_progress_summary()
        
        print("🔍 EXPERIMENT STATUS")
        print("="*40)
        print(f"Running: {'✅ YES' if status['is_running'] else '❌ NO'}")
        print(f"PID: {status['pid'] or 'N/A'}")
        print(f"Progress: {progress['overall_progress']:.1f}%")
        print(f"Start Time: {status['start_time'] or 'N/A'}")
        
        if status['errors']:
            print(f"\n⚠️  Recent Errors ({len(status['errors'])}):")
            for error in status['errors'][-3:]:
                print(f"  - {error}")
    
    def generate_report(self):
        """Generate status report"""
        report = self.monitor.create_status_report()
        print(report)
    
    def create_plots(self):
        """Create progress plots"""
        self.monitor.create_progress_plot('results/progress_plot.png')
        print("📊 Progress plot saved to results/progress_plot.png")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Weekly Breakout Prediction Experiment')
    parser.add_argument('--config', default='config/experiment_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--background', action='store_true', 
                       help='Run in background mode')
    parser.add_argument('--foreground', action='store_true', 
                       help='Run in foreground mode (internal)')
    parser.add_argument('--check', action='store_true', 
                       help='Check experiment status')
    parser.add_argument('--report', action='store_true', 
                       help='Generate status report')
    parser.add_argument('--plot', action='store_true', 
                       help='Create progress plots')
    parser.add_argument('--monitor', action='store_true', 
                       help='Start continuous monitoring')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    
    if args.check:
        runner.check_status()
    elif args.report:
        runner.generate_report()
    elif args.plot:
        runner.create_plots()
    elif args.monitor:
        runner.monitor.monitor_continuously(args.interval)
    elif args.foreground:
        runner._run_foreground()
    else:
        runner.run_experiment(background=args.background)

if __name__ == "__main__":
    main()
