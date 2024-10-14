import pstats
file_name = ['output_file', 'output_file2']
for fname in file_name:
    p = pstats.Stats(fname)
    p.sort_stats('cumulative').print_stats(10)

