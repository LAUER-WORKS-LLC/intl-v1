"""
Experiment Monitoring System
Provides real-time monitoring and status checking for unattended runs
"""

import os
import json
import time
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentMonitor:
    """Monitor experiment progress and health"""
    
    def __init__(self, experiment_dir: str = "."):
        self.experiment_dir = experiment_dir
        self.log_file = os.path.join(experiment_dir, "results", "run.log")
        self.status_file = os.path.join(experiment_dir, "results", "status.json")
        self.progress_file = os.path.join(experiment_dir, "results", "progress.json")
        
    def check_experiment_status(self) -> Dict:
        """Check if experiment is running and get status"""
        status = {
            'is_running': False,
            'pid': None,
            'start_time': None,
            'progress': 0,
            'current_phase': None,
            'last_update': None,
            'errors': []
        }
        
        try:
            # Check if process is running
            result = subprocess.run(['pgrep', '-f', 'weekly_breakout_experiment.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                status['is_running'] = True
                status['pid'] = result.stdout.strip()
                
                # Get process info
                try:
                    pid = int(status['pid'])
                    process = psutil.Process(pid)
                    status['start_time'] = datetime.fromtimestamp(process.create_time())
                    status['cpu_percent'] = process.cpu_percent()
                    status['memory_mb'] = process.memory_info().rss / 1024 / 1024
                except:
                    pass
            
            # Check progress
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    status.update(progress_data)
            
            # Check for errors
            if os.path.exists(self.log_file):
                status['errors'] = self._check_for_errors()
            
            # Update status
            status['last_update'] = datetime.now().isoformat()
            
        except Exception as e:
            status['errors'].append(f"Error checking status: {str(e)}")
        
        return status
    
    def _check_for_errors(self) -> List[str]:
        """Check log file for errors"""
        errors = []
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
            # Look for error patterns in last 100 lines
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            for line in recent_lines:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                    errors.append(line.strip())
                    
        except Exception as e:
            errors.append(f"Error reading log file: {str(e)}")
        
        return errors
    
    def get_progress_summary(self) -> Dict:
        """Get detailed progress summary"""
        summary = {
            'overall_progress': 0,
            'phases': {},
            'data_status': {},
            'results_status': {}
        }
        
        try:
            # Check data download progress
            data_dir = os.path.join(self.experiment_dir, "data", "prices_daily")
            if os.path.exists(data_dir):
                ticker_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
                summary['data_status']['tickers_downloaded'] = len(ticker_files)
                summary['data_status']['download_progress'] = len(ticker_files) / 500 * 100  # Assuming 500 tickers
            
            # Check feature computation
            features_file = os.path.join(self.experiment_dir, "artifacts", "features.parquet")
            if os.path.exists(features_file):
                df = pd.read_parquet(features_file)
                summary['data_status']['features_computed'] = len(df)
                summary['phases']['feature_engineering'] = 100
            
            # Check labels
            labels_file = os.path.join(self.experiment_dir, "artifacts", "labels.parquet")
            if os.path.exists(labels_file):
                df = pd.read_parquet(labels_file)
                summary['data_status']['labels_generated'] = len(df)
                summary['phases']['label_generation'] = 100
            
            # Check CV setup
            folds_file = os.path.join(self.experiment_dir, "cv", "folds.csv")
            if os.path.exists(folds_file):
                folds_df = pd.read_csv(folds_file)
                summary['data_status']['cv_folds'] = len(folds_df)
                summary['phases']['cv_setup'] = 100
            
            # Check hyperparameter search
            val_results_file = os.path.join(self.experiment_dir, "metrics", "val", "metrics_by_config.csv")
            if os.path.exists(val_results_file):
                results_df = pd.read_csv(val_results_file)
                summary['results_status']['configurations_tested'] = len(results_df)
                summary['phases']['hyperparameter_search'] = min(100, len(results_df) / 1000 * 100)
            
            # Calculate overall progress
            phase_progress = list(summary['phases'].values())
            if phase_progress:
                summary['overall_progress'] = sum(phase_progress) / len(phase_progress)
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def create_status_report(self) -> str:
        """Create human-readable status report"""
        status = self.check_experiment_status()
        progress = self.get_progress_summary()
        
        report = f"""
🔍 EXPERIMENT STATUS REPORT
{'='*50}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATUS:
{'✅ RUNNING' if status['is_running'] else '❌ NOT RUNNING'}
PID: {status['pid'] or 'N/A'}
Start Time: {status['start_time'] or 'N/A'}
Progress: {progress['overall_progress']:.1f}%

📈 PHASE PROGRESS:
"""
        
        for phase, progress_pct in progress['phases'].items():
            report += f"  {phase.replace('_', ' ').title()}: {progress_pct:.1f}%\n"
        
        report += f"""
📁 DATA STATUS:
  Tickers Downloaded: {progress['data_status'].get('tickers_downloaded', 0)}
  Features Computed: {progress['data_status'].get('features_computed', 0)}
  Labels Generated: {progress['data_status'].get('labels_generated', 0)}
  CV Folds: {progress['data_status'].get('cv_folds', 0)}

🔬 RESULTS STATUS:
  Configurations Tested: {progress['results_status'].get('configurations_tested', 0)}
"""
        
        if status['errors']:
            report += f"""
⚠️  RECENT ERRORS:
"""
            for error in status['errors'][-5:]:  # Last 5 errors
                report += f"  - {error}\n"
        
        return report
    
    def save_status(self):
        """Save current status to file"""
        status = self.check_experiment_status()
        progress = self.get_progress_summary()
        
        status_data = {
            'status': status,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def create_progress_plot(self, save_path: str = None):
        """Create progress visualization"""
        try:
            # Load historical progress if available
            progress_data = []
            
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
            
            if not progress_data:
                return
            
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Overall progress over time
            if 'timestamps' in progress_data and 'overall_progress' in progress_data:
                axes[0, 0].plot(progress_data['timestamps'], progress_data['overall_progress'])
                axes[0, 0].set_title('Overall Progress Over Time')
                axes[0, 0].set_ylabel('Progress (%)')
            
            # Phase completion
            if 'phases' in progress_data:
                phases = list(progress_data['phases'].keys())
                completion = list(progress_data['phases'].values())
                axes[0, 1].bar(phases, completion)
                axes[0, 1].set_title('Phase Completion')
                axes[0, 1].set_ylabel('Completion (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Data status
            if 'data_status' in progress_data:
                data_items = list(progress_data['data_status'].keys())
                data_values = list(progress_data['data_status'].values())
                axes[1, 0].bar(data_items, data_values)
                axes[1, 0].set_title('Data Status')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Results status
            if 'results_status' in progress_data:
                results_items = list(progress_data['results_status'].keys())
                results_values = list(progress_data['results_status'].values())
                axes[1, 1].bar(results_items, results_values)
                axes[1, 1].set_title('Results Status')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating progress plot: {str(e)}")
    
    def monitor_continuously(self, interval: int = 300):
        """Monitor experiment continuously"""
        print(f"🔍 Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Check status
                status = self.check_experiment_status()
                progress = self.get_progress_summary()
                
                # Print status
                print(f"\n{datetime.now().strftime('%H:%M:%S')} - Status: {'RUNNING' if status['is_running'] else 'STOPPED'}")
                print(f"Progress: {progress['overall_progress']:.1f}%")
                
                if status['errors']:
                    print(f"Errors: {len(status['errors'])}")
                
                # Save status
                self.save_status()
                
                # Wait
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor weekly breakout experiment')
    parser.add_argument('--check', action='store_true', help='Check status once')
    parser.add_argument('--monitor', action='store_true', help='Monitor continuously')
    parser.add_argument('--report', action='store_true', help='Generate status report')
    parser.add_argument('--plot', action='store_true', help='Create progress plot')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor()
    
    if args.check:
        status = monitor.check_experiment_status()
        print(json.dumps(status, indent=2))
    
    if args.report:
        report = monitor.create_status_report()
        print(report)
    
    if args.plot:
        monitor.create_progress_plot('results/progress_plot.png')
    
    if args.monitor:
        monitor.monitor_continuously(args.interval)

if __name__ == "__main__":
    main()
