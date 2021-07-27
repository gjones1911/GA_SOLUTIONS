import os
import sys




def test_runs(exe_file, numruns=1, r1=None, r2=None):
    """
        Will run the given python command on the command line in the form
            * python exe_file
        if the numruns is not None,otherwise it will run the given
        command of the form:
            * python exe_file num
        where num is the numbers form r1 to r2-1
    :param exe_file: string representing the command to give to the command line
    :param numruns:
    :param r1:
    :param r2:
    :return:
    """
    if numruns is not None:
        for i in range(numruns):
            os.system('python {}'.format(exe_file))
    else:
        print('range')
        for i in range(r1,r2):
            print('python {}'.format(exe_file + ' ' + str(i)))
            os.system('python {}'.format(exe_file + ' '+ str(i)))


def iterative_tester_cmd_interface(argv, verbose=False):
    numruns = int(argv[1])
    #   0        1     2   3 ....
    # exename numruns exe args.....
    cmd = argv[2:]
    filetorun = ''
    for i in range(len(cmd)):
        if verbose:
            print(cmd[i])
        filetorun += cmd[i] + ' '
    if verbose:
        print('Running {} trails'.format(numruns))
        print('Given {} runs to test'.format(numruns))
        print('the command {}'.format(cmd))
        print('running the command {}'.format(filetorun))
    return numruns, filetorun


def interactive_iterative_tester_interface():
    filetorun = input('Give me the name of the executable you want to run\n'
                      'and any command lines needed seperated by spaces: ')
    filetorun = filetorun.split(' ')
    filetorunfinal = ""
    for c in filetorun:
        c.strip()
        filetorunfinal += c + " "
    filetorun = filetorunfinal
    print('running the command {}'.format(filetorunfinal))
    numruns = int(input('How many test runs would you like?: '))
    return numruns, filetorun