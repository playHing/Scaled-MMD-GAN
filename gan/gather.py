import shutil, os
from glob import glob
import sys

path = sys.argv[1]
dir_ = '/'.join(path.split('/')[:-3])
files = ['/'.join(f.split('/')[-3:]) for f in glob(path)]
print('directory: ' + dir_ )
ndir = os.path.join(dir_, 'all_results')
if not os.path.exists(ndir):
    os.makedirs(ndir)
for f in files:
    origin = os.path.join(dir_, f)
    dest = os.path.join(ndir, f.replace('/', '-'))
    shutil.copyfile(origin, dest)
    print('FILE ' + origin + '\nCOPIED TO ' + dest + '\n')



