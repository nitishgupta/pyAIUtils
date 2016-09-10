import tensorflow as tf


def print_var_list(var_list, name='Variables'):
    print name + ': \n' + '[' + ', '.join([var.name for var in var_list]) + ']'


def collect_name(var_name, graph=None, var_type=tf.GraphKeys.VARIABLES):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(var_type, scope=var_name)

    assert_str = "No variable exists with name '{}'".format(var_name)
    assert len(var_list) != 0, assert_str

    assert_str = \
        "Multiple variables exist with name_scope '{}'".format(var_name)
    assert len(var_list) == 1, assert_str

    return var_list[0]


def collect_scope(name_scope, graph=None, var_type=tf.GraphKeys.VARIABLES):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(var_type, scope=name_scope)

    assert_str = "No variable exists with name_scope '{}'".format(name_scope)
    assert len(var_list) != 0, assert_str

    return var_list


def collect_all(graph=None):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(tf.GraphKeys.VARIABLES)

    return var_list

def get_all_scopes_in_var(var_name):
    # Split at ':' to remove the variable number at the end
    wo_var_num = var_name.split(":")[0]

    # Split at '/' to get all scope names and variable name
    scopes_names = wo_var_num.split("/")

    return set(scopes_names)


def collect_partial_scope(
    name_scope,
    graph=None,
    var_type=tf.GraphKeys.VARIABLES):
    ''' Function for collecting variables that contain the given name_scope in
    their scope hierarchy.
    Eg. If var has var.name = 'scope1/scope2/scope3/var_name:4'
    Then name_scope='scope2' will collect this variable (if it exists in
    var_type collection)
    name_scope cannot contain moer than 1 scope. Eg. It cannot be
    'scope2/scope3'
    '''
    if graph == None:
        graph = tf.get_default_graph()
    var_list = graph.get_collection(var_type)

    scope_var_list = []
    for var in var_list:
      var_scope_names = self.get_all_scopes_in_var(var.name)
      if name_scope in var_scope_names:
        scope_var_list.append(var)

    assert_str = "No variable exists with name_scope '{}'".format(name_scope)
    assert len(scope_var_list) != 0, assert_str

    return scope_var_list


def collect_all_trainable(graph=None):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    return var_list


def collect_list(var_name_list, graph=None, var_type=tf.GraphKeys.VARIABLES):
    var_dict = dict()
    for var_name in var_name_list:
        var_dict[var_name] = collect_name(
            var_name, graph=graph, var_type=var_type)

    return var_dict
