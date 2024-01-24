
import os, sys
from pathlib import Path
here = Path(__file__).parent.absolute()

sys.path.append(f'{here}/mmdetection')

from mmdetection.tools.train import main

if __name__ == '__main__':
    main()