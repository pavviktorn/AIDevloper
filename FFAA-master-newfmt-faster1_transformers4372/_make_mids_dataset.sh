#!/bin/bash

export PYTHONPATH=/datasets/work/vLLM/FFAA-master-newfmt:$PYTHONPATH
python make_mids_dataset.py --which_part 1 &
python make_mids_dataset.py --which_part 2 &
python make_mids_dataset.py --which_part 3 &
python make_mids_dataset.py --which_part 4 &
python make_mids_dataset.py --which_part 5 &
python make_mids_dataset.py --which_part 6 &
python make_mids_dataset.py --which_part 7 &

# Wait for both to finish
wait
echo "All parts finished."
