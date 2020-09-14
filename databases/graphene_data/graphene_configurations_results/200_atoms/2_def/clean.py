import os
import re
from copy import copy

files = os .listdir()

for f in files:
    new_name = copy(f)
    re.sub("(|)","", new_name)
    os.rename(f, new_name)
