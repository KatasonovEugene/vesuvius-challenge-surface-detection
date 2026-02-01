import subprocess

from src.utils.io_utils import ROOT_PATH

path_to_repo = ROOT_PATH / 'ext' / 'vesuvius_metric_resources' / 'topological_metrics_kaggle'
path_to_build1 = path_to_repo / 'external' / 'Betti-Matching-3D' / 'build'

path_to_build1.mkdir(parents=True, exist_ok=True)
subprocess.run(['cmake', '..'], cwd=path_to_build1, check=True)
subprocess.run(['make'], cwd=path_to_build1, check=True)

subprocess.run(['make', 'build-betti'], cwd=path_to_repo, check=True)
subprocess.run(['make', 'dev'], cwd=path_to_repo, check=True)
