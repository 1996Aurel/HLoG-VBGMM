import argparse

parser = argparse.ArgumentParser(description="HLoG Blob detector arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("sample", type=str, help="A, B, C, D, E or F.")
args = parser.parse_args()
