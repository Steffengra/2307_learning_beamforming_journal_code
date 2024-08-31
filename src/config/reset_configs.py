
from shutil import copyfile
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))


if __name__ == '__main__':
    for file in [
        'config',
        'config_error_model',
        'config_sac_learner',
    ]:
        copyfile(
            src=Path(project_root_path, 'src', 'config', f'{file}.py.default'),
            dst=Path(project_root_path, 'src', 'config', f'{file}.py'),
        )
