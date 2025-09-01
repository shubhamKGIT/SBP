
import argparse
from sbputils.sbp import SBP, Pyrodata
import pathlib

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Call main for SBP algo with reference to experiment files")
    argparser.add_argument("--exp", 
                           type= int,
                           default= 12,
                            help="giv three digit experiment number, say 012, inputs as string")
    my_Exp = SBP(myExperiment=Pyrodata(exp_number=12,
                                       basepath=pathlib.Path(__file__).parent,
                                       filenames=None    # it will jsut pick the mraw awaviable
                                       )
                )
    my_Exp.data_holder.read_spectral_data()
    my_Exp.plot_raw_spectra()