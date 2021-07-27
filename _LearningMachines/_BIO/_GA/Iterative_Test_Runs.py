from _scripting_methods import *
from _utils import *
import sys

usage_msg = "Usage: python test_runs_test.py number_of_runs exe.py cmd1 cmd2 .....cmdN\n" +\
            "or,\n"+\
            "Usage: python test_runs_test.py r start end exe.py cmd1 cmd2 .....cmdN"
break_dict = {
                "eq":1,
}

arg_length_check(sys.argv, break_dict, usage_msg=usage_msg)

numruns, cmd = iterative_tester_cmd_interface(sys.argv, verbose=False)

test_runs(cmd, numruns=numruns)