import os
import ase
import sys
from ase.db import connect
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms

class DbCreator:
    def __init__(self, dbname):
        self.dbname = dbname

    def create(self):
        files = [f for f in os.listdir() if f.endswith(".xyz")]
        with connect(self.dbname) as db:
            for f in files:
                posinp = Posinp.from_file(f)
                with open(f, "r") as posinp_file:
                    energy = float(posinp_file.readline().split()[2]) * 27.21138602
                    forces = None
                    for line in posinp_file:
                        if "forces" not in line:
                            continue
                        else:
                            forces = []
                            for _ in range(len(posinp)):
                                forces.append(
                                    [
                                        float(force) * 27.21138602 / 0.529177249
                                        for force in posinp_file.readline().split()[-3:]
                                    ]
                                )
                            break
                atoms = posinp_to_ase_atoms(posinp)
                db.write(atoms, data={"energy": energy, "forces": forces})

    @property
    def dbname(self):
        return self._dbname

    @dbname.setter
    def dbname(self, dbname):
        dbname = str(dbname)
        if dbname.endswith(".db"):
            self._dbname = dbname
        else:
            self._dbname = dbname + ".db"


if __name__ == "__main__":
    dbc = DbCreator(name=sys.argv[1])
    dbc.create()
