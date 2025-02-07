# %%
import logging
from pathlib import Path
import re
import sys
import shutil

# %%
projroot = Path('..')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
regex_pattern = (r".*Subject ([^\.]+)\.txt")
if not len(sys.argv) == 5:
    logging.error('Incorrect number of arguments, use\n'
                  'python move_and_label_medpc.py <sourcepath> <ses> <task> <acq>')
    sys.exit(1)
sourcepath = Path(sys.argv[1])
ses, task, acq = sys.argv[2:5]
if not sourcepath.exists():
    logging.error(f'Provided source path {sourcepath} does not exist')
    sys.exit(1)
for fn in sourcepath.glob('*Subject *.txt'):
    match = re.search(regex_pattern, str(fn.name))
    if not match:
        logging.warning(f'Bad template match for {fn.name}')
        continue
    sub = match.groups()[0].lower()
    med_fn = (projroot / 'rawdata' / f'sub-{sub}' / f'ses-{ses}' /
              f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_rawmed.txt')
    if med_fn.exists():
        logging.info(f'Raw med events file already exists for {fn.name}')
        continue
    print(f'Renaming {fn.absolute()} to {med_fn.absolute()}')
    med_fn.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fn, med_fn)


