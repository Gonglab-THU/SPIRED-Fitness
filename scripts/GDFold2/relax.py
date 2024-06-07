import timeit
import argparse
from Bio import SeqIO
from pyrosetta import *

parser = argparse.ArgumentParser(description="Perform FastRelax on predicted protein structure")
parser.add_argument("--input", type=str, help="input unrelaxed model")
parser.add_argument("--output", type=str, help="output model (.pdb format)")
parser.add_argument("--repeat", type=int, default=2, help="number of repeats, default=2")
parser.add_argument("--cycle", type=int, default=200, help="number of max cycles, default=200")
args = parser.parse_args()


def initialize(repeat=2, cycle=200):
    init_cmd = list()
    init_cmd.append("-mute all")
    init_cmd.append(f"-relax:default_repeats {repeat}")
    init_cmd.append(f"-default_max_cycles {cycle}")
    init_cmd.append("-relax:constrain_relax_to_start_coords")
    init_cmd.append("-relax:ramp_constraints false")
    init_cmd.append("-relax:dualspace true -relax::minimize_bond_angles")
    init_cmd.append("-relax:jump_move true -relax:bb_move true -relax:chi_move true")
    init(" ".join(init_cmd))


def fastrelax(input, output):
    pose = pose_from_file(input)
    scorefxn = create_score_function("ref2015_cart")
    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
    fr.apply(pose)
    pose.dump_pdb(output)


if __name__ == "__main__":
    for i in SeqIO.parse(args.input, "fasta"):
        name = i.description
        start_time = timeit.default_timer()
        initialize(args.repeat, args.cycle)
        print("Relaxing...")
        fastrelax(f"{args.output}/{name}/GDFold2/fold_0.pdb", f"{args.output}/{name}/relax.pdb")
        end_time = timeit.default_timer()
        print("Running time: {:.2f}s".format(end_time - start_time))
