
import sys
import operator
# ##################################################
# ##################################################
# ##################################################
# TODO: quick_logic
# ##################################################
# ##################################################
# ##################################################

def q_logic(typ, vaA, vaB):
    options_l =['le', 'leq', 'eq', 'geq', 'ge', 'neq', 'n_in', '_in', 'and', '_is']
    if typ not in options_l:
        usage('Unknown type: {}\noptions: {}'.format(typ, options_l))
        return None
    elif typ == 'le':
        return vaA < vaB
    elif typ == 'leq':
        return vaA <= vaB
    elif typ == 'eq':
        return vaA == vaB
    elif typ == 'geq':
        return vaA >= vaB
    elif typ == 'ge':
        return vaA > vaB
    elif typ == 'neq':
        return vaA != vaB
    elif typ == 'n_in':
        return vaA not in vaB
    elif typ == '_in':
        return vaA in vaB
    elif typ == '_is':
        return vaA is vaB
    elif typ == '_is_not':
        return vaA is vaB
    elif typ == 'and':
        return vaA and vaB


def sort_dict(dic, sort_by='vals', reverse=False):
    """
        Returns a sorted version of the given dictionary
    :param dic: dictionary to sort
    :param sort_by: 'vals' to sort by values, 'keys', to sort by keys
    :param reverse: set to to True to get largest to smallest
    :return:
    """
    if sort_by == 'vals':
        return dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=reverse))
    elif sort_by == 'keys':
        return dict(sorted(dic.items(), key=operator.itemgetter(0), reverse=reverse))


# ##################################################
# ##################################################
# ##################################################
# TODO: User messaging
# ##################################################
# ##################################################
# ##################################################

def usage(usage_str):
    print(usage_str)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def stderr(msg, sep=' ', msg_num=-99):
    eprint(msg, sep=sep)
    quit(msg_num)

# ##################################################
# ##################################################
# ##################################################
# TODO: Command line handling
# ##################################################
# ##################################################
# ##################################################

def arg_length_check(argv, break_dict=None, usage_msg=None):
    """
                Checks the given command line vector for the appropriate length using the break dict
                to check for break conditions
    :param argv: sys.argv
    :param break_dict: dictionary of form {
                                            "condition":value    checking that len(argv) "condition" value is not true
                                          }
                        where "condition" is one of:
                             ['le' (<),
                              'leq'(<=),
                              'eq'(==),
                              'geq'(>=),
                              'ge'(>),
                              'neq'(!=),
                              'n_in'( not in),
                              '_in'(in)]
    :param usage_msg: a message to display when the error occurs
    :return:
    """
    if q_logic('_is', break_dict, None):
        break_dict = {
                        "neq":1,
        }
    if q_logic('_is', usage_msg, None):
        # basic message for no command line arguments
        usage_msg = 'There must {} argument'.format(0)
    for v in break_dict:
        reps = q_logic(v, len(argv), break_dict[v])
        if reps is None:
            quit(-70)
        if reps is True:
            usage(usage_msg)
            quit(-70)

    return


def basic_cmd(argv):
    """
        returns a list of the command line arguments
    """
    return [ argv[i] for i in range(1, len(argv))]



def handle_command_args(argv, handeler, break_dict=None, usage_msg=None):
    #check the command line
    arg_length_check(argv, break_dict, usage_msg=usage_msg)
    return handeler(argv)