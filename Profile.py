# python3 Profile.py <profile_dump_file> > <out_file>
import sys
import pstats

if __name__ == "__main__" : 
    prIn = sys.argv[1]
    stats = pstats.Stats(prIn)
    stats = stats.sort_stats('cumulative')
    stats.print_stats()

