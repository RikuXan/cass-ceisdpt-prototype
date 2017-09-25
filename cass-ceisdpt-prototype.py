import itertools
import argparse
import os.path
import math
import scipy.stats
from interval import interval
import json
import ujson as ujson


# returns a deep copy (i.e. no references) to the provided dictionary, faster than copy.deepcopy
def deepcopy_dictionary(dictionary):
    return ujson.loads(ujson.dumps(dictionary))


# Calculate the probability that the condition of a disruptor takes effect
# for performance reasons, check if that exact same disruptor has already been calculated and if so, load the result from cache (disruptor_probability_map)
disruptor_probability_map = {}
def calculate_disruptor_probability(disruptor):
    disruptor_id = get_dict_id(disruptor)

    if disruptor_id in disruptor_probability_map:
        return disruptor_probability_map[disruptor_id]

    referenced_xcv = g_xcvs[disruptor['xcv']]
    disruptor_probability = 0

    for condition in disruptor['conditions']:
        if referenced_xcv['type'] == 'discrete_values':
            disruptor_probability += referenced_xcv['data'][condition]

        elif referenced_xcv['type'] == 'continuous_distribution':
            if referenced_xcv['data']['distribution'] == 'normal':
                dist = scipy.stats.norm(referenced_xcv['data']['mean'], math.sqrt(referenced_xcv['data']['standard_deviation']))
            elif referenced_xcv['data']['distribution'] == 'uniform':
                dist = scipy.stats.uniform(referenced_xcv['data']['left'], referenced_xcv['data']['right'])
            elif referenced_xcv['data']['distribution'] == 'exponential':
                dist = scipy.stats.uniform(1 / referenced_xcv['data']['rate'])
            else:
                raise AttributeError("XCV distribution type is not in ['normal']")

            disruptor_probability += dist.cdf(condition[1]) - dist.cdf(condition[0])

        else:
            raise AttributeError("XCV type is not in ['discrete_values', 'continuous_distribution']")

    disruptor_probability_map[disruptor_id] = disruptor_probability
    return disruptor_probability


# return a hash based on a dictionary's contents (same contents = same hash)
def get_dict_id(dictionary):
    return hash(json.dumps(dictionary, sort_keys=True))


# calculate the aggregate nfp values over a composition
def calculate_composition_aggregates(service_composition):
    composition_total_cost = 0.0
    composition_total_type_score = 0.0
    composition_actual_duration = 0.0
    composition_total_duration = 0.0

    if service_composition['status'] != 'Valid':
        return

    for service in service_composition['services']:
        # do not consider service whose execution didn't even start
        if service['execution_percentage'] == 0.0:
            continue

        service_actual_duration = g_service_data[service['name']]['avg_duration'] * service['execution_percentage']

        # cost (sum)
        service_actual_cost = g_service_data[service['name']]['avg_price']
        for discount_to_apply in filter(lambda discount: discount['service'] == service['name'],
                                        service['state_before']['discounts']):
            service_actual_cost -= discount_to_apply['discount_value']

        composition_total_cost += service_actual_cost

        # type score (weighted sum)
        composition_total_type_score += g_service_data[service['name']]['type_score'] * service_actual_duration

        # duration
        composition_actual_duration += service_actual_duration
        composition_total_duration += g_service_data[service['name']]['avg_duration']

    service_composition['total_cost'] = composition_total_cost
    service_composition['total_type_score'] = composition_total_type_score / composition_actual_duration
    service_composition['average_execution_percentage'] = composition_actual_duration / composition_total_duration


# Calculate the utility of a composition, based on its end-to-end NFPs and the SAW method
def calculate_composition_utility(service_composition, aggregates_extreme_values):
    composition_utility = 0.0

    if service_composition['status'] != 'Valid':
        service_composition['utility'] = 0
        return

    for utility_factor_name, utility_factor in g_utility_factors.items():
        utility_factor_value = service_composition[utility_factor_name] * (1 if utility_factor['type'] == 'min' else -1)

        # max and min value for the utility factor are the same, according to the equation, this adds 0/0 * w_k to the utility value, as this is undefined we replace this with 1
        if not aggregates_extreme_values[utility_factor_name]['max'] > aggregates_extreme_values[utility_factor_name]['min']:
            composition_utility += 1 * utility_factor['weight']
        else:
            composition_utility += (aggregates_extreme_values[utility_factor_name]['max'] - utility_factor_value) / \
                                   (aggregates_extreme_values[utility_factor_name]['max'] - aggregates_extreme_values[utility_factor_name]['min']) * utility_factor['weight']

    service_composition['utility'] = composition_utility


# calculate the expected utility (EU) of a composition, based on its and its subcompositions utilities and execution_probabilities
# traverse depth-first (head recursion) through the alternative composition candidates and calculate the EU for the deepest candidates (i.e. the ones without alternative compositions) first
# then choose the one with the highest EU as the definitive alternative composition to use for further calculations
def calculate_composition_expected_utility(service_composition, aggregates_extreme_values):
    for interruption_possibility in service_composition['interruption_possibilities']:
        for alternative_composition_candidate in interruption_possibility['alternative_composition_candidates']:
            calculate_composition_expected_utility(alternative_composition_candidate, aggregates_extreme_values)

        if interruption_possibility['alternative_composition_candidates']:
            interruption_possibility['alternative_composition'] = max(interruption_possibility['alternative_composition_candidates'], key=lambda acc: acc['expected_utility'])

    expected_utility = 0.0
    total_execution_probability = 0.0
    calculate_composition_utility(service_composition, aggregates_extreme_values)

    service_composition_subcompositions = get_self_and_subcompositions(service_composition)
    for subcomposition in service_composition_subcompositions:
        expected_utility += subcomposition['utility'] * subcomposition['execution_probability']
        total_execution_probability += subcomposition['execution_probability']

    service_composition['expected_utility'] = expected_utility / total_execution_probability


# return a list composed of the provided composition and all of its subcompositions
def get_self_and_subcompositions(service_composition):
    service_composition_subcompositions = [service_composition]

    for interruption_possibility in service_composition['interruption_possibilities']:
        service_composition_subcompositions.extend(get_self_and_subcompositions(interruption_possibility['alternative_composition']))

    return service_composition_subcompositions


# return a list composed of the provided composition and all of its subcomposition candidates
def get_self_and_subcomposition_candidates(service_composition):
    service_composition_subcompositions = [service_composition]

    for interruption_possibility in service_composition['interruption_possibilities']:
        for alternative_composition_candidate in interruption_possibility['alternative_composition_candidates']:
            service_composition_subcompositions.extend(get_self_and_subcomposition_candidates(alternative_composition_candidate))

    return service_composition_subcompositions


# return a list of all possible compositions based on the list of global compositions given by the full enumeration
def get_possible_service_compositions(service_compositions):
    possible_service_compositions = []

    for service_composition in service_compositions:
        possible_service_compositions.extend(get_self_and_subcomposition_candidates(service_composition))

    return possible_service_compositions

# returns a dictionary with the minimum and maximum values for each defined utility factor in the provided compositions
# best to provide all possible compositions for the problem in order to correctly set the SAW boundaries
def get_aggregates_extreme_values(service_compositions):
    aggregates_extreme_values = {}

    for service_composition in service_compositions:
        if service_composition['status'] != 'Valid':
            continue

        for utility_factor_name, utility_factor in g_utility_factors.items():
            if utility_factor_name not in aggregates_extreme_values:
                aggregates_extreme_values[utility_factor_name] = {}

            utility_factor_value = service_composition[utility_factor_name] * (1 if utility_factor['type'] == 'min' else -1)

            if 'min' not in aggregates_extreme_values[utility_factor_name] or \
               aggregates_extreme_values[utility_factor_name]['min'] > utility_factor_value:
                aggregates_extreme_values[utility_factor_name]['min'] = utility_factor_value

            if 'max' not in aggregates_extreme_values[utility_factor_name] or \
               aggregates_extreme_values[utility_factor_name]['max'] < utility_factor_value:
                aggregates_extreme_values[utility_factor_name]['max'] = utility_factor_value

    return aggregates_extreme_values


# caclculates the conditional probability for a possible replacements state, given the supplied interruption context
def calculate_replacement_state_conditional_probability(replacement_state, interruption_context):
    replacement_state_conditional_probability = 0

    # merge the disruptors of our replacement state by xcv
    merged_replacement_disruptors = []
    replacement_disruptors = sorted([disruptor_state['disruptor'] if disruptor_state['active'] else
                                     get_complementary_disruptor(disruptor_state['disruptor']) for disruptor_state in replacement_state],
                                    key=lambda dis: dis['xcv'])
    for xcv, replacement_disruptor_group in itertools.groupby(replacement_disruptors, key=lambda dis: dis['xcv']):
        merged_replacement_disruptors.append(get_intersected_disruptors(*replacement_disruptor_group))

    for interruption_state in interruption_context:
        intersection_probability = 1
        for disruptor_state in interruption_state['state']:
            relevant_interruption_state_disruptor = disruptor_state['disruptor'] if disruptor_state['active'] else get_complementary_disruptor(disruptor_state['disruptor'])
            try:
                related_replacement_disruptor = next(filter(lambda dis: dis['xcv'] == disruptor_state['disruptor']['xcv'], merged_replacement_disruptors))
            except StopIteration:
                continue

            interruption_replacement_intersection = get_intersected_disruptors(relevant_interruption_state_disruptor, related_replacement_disruptor)
            intersection_probability *= calculate_disruptor_probability(interruption_replacement_intersection) / calculate_disruptor_probability(relevant_interruption_state_disruptor)

        replacement_state_conditional_probability += intersection_probability * interruption_state['probability']

    return replacement_state_conditional_probability / sum(istate['probability'] for istate in interruption_context)


# returns a virtual disruptor with the complementary condition to the original disruptor
def get_complementary_disruptor(disruptor):
    xcv_type = g_xcvs[disruptor['xcv']]['type']
    if xcv_type == 'discrete_values':
        condition_complement = list(set(g_xcvs[disruptor['xcv']]['data'].keys()) - set(disruptor['conditions']))
    elif xcv_type == 'continuous_distribution':
        condition_complement = get_interval_complement(disruptor['conditions'])
    else:
        raise AttributeError("disruptor['type'] no in ['discrete_values', 'continuous_distribution']")

    return {
        'xcv': disruptor['xcv'],
        'conditions': condition_complement
    }


# returns a virtual disruptor with the intersection of the provided disruptors' conditions
def get_intersected_disruptors(*args):
    xcv_type = g_xcvs[args[0]['xcv']]['type']
    for disruptor in args[1:]:
        if g_xcvs[disruptor['xcv']]['type'] != xcv_type:
            raise AttributeError("Disruptor types are not equal")

    if xcv_type == 'discrete_values':
        condition_intersection = set(args[0]['conditions'])
    elif xcv_type == 'continuous_distribution':
        condition_intersection = args[0]['conditions']
    else:
        raise AttributeError("disruptor['type'] no in ['discrete_values', 'continuous_distribution']")

    for disruptor in args[1:]:
        if xcv_type == 'discrete_values':
            condition_intersection = condition_intersection & set(disruptor['conditions'])
        elif xcv_type == 'continuous_distribution':
            condition_intersection = condition_intersection & disruptor['conditions']

    if xcv_type == 'discrete_values':
        condition_intersection = list(condition_intersection)

    return {
        'xcv': args[0]['xcv'],
        'conditions': condition_intersection
    }


# calculates the complement to a numerical interval
def get_interval_complement(iv):
    chain = itertools.chain(iv, [[math.inf, None]])
    out = []

    prev = [None, -math.inf]
    for this in chain:
        if prev[1] != this[0]:
            out.append([prev[1], this[0]])
        prev = this

    return interval(*out)


# calculates p_intsingle (i.e. the probability that a service gets interrupted based on its disruptors in a single timeframe)
service_name_interruption_probability_map = {}
def calculate_service_interruption_probability(service):
    if service['name'] in service_name_interruption_probability_map:
        return service_name_interruption_probability_map[service['name']]

    execution_probability = 1
    for interruption_disruptor in g_service_data[service['name']]['disruptors']:
        execution_probability *= 1 - calculate_disruptor_probability(interruption_disruptor)

    service_name_interruption_probability_map[service['name']] = 1 - execution_probability
    return service_name_interruption_probability_map[service['name']]


# returns the blacklist for replacement services based on the interrupted_service and previous_blacklist (the services that need not be considered for the blacklist due to them not being considered for further compositions anyways)
def get_replacement_blacklist(interrupted_service, previous_blacklist):
    replacement_blacklist = []

    possible_replacement_services = list(filter(lambda replacement_service: replacement_service['service_class'] == g_service_data[interrupted_service['name']]['service_class'] and
                                                                            replacement_service['name'] != interrupted_service['name'] and
                                                                            replacement_service['name'] not in previous_blacklist, g_services))
    replacement_disruptors = list(filter(lambda replacement_disruptor: replacement_disruptor['xcv'] in
                                                                                [disruptor['xcv'] for disruptor in g_service_data[interrupted_service['name']]['disruptors']],
                                                  itertools.chain.from_iterable([possible_replacement_service['disruptors'] for possible_replacement_service in possible_replacement_services])))

    if len(replacement_disruptors) == 0:
        return [
            {
                'blacklisted_services': set(),
                'probability': 1
            }
        ]

    # build interruption context from the disruptors of the interrupted service
    interruption_context = [
        {
            'state': interruption_state,
            'probability': scipy.prod([calculate_disruptor_probability(disruptor_state['disruptor']) if disruptor_state['active'] else \
                                                                       1 - calculate_disruptor_probability(disruptor_state['disruptor']) for disruptor_state in interruption_state])
        } for interruption_state in
              list(itertools.product(*[
                  [
                      {
                          'disruptor': disruptor,
                          'active': True
                      },
                      {
                          'disruptor': disruptor,
                          'active': False
                      }
                  ]
                  for disruptor in g_service_data[interrupted_service['name']]['disruptors']]))[:-1]] # cut off last element (no interruption)

    # build replacement context from all eligible disruptors (possible replacement services, referenced xcv also in interrupted service's disruptors
    replacement_context = [
        {
            'state': replacement_state,
            'probability': calculate_replacement_state_conditional_probability(replacement_state, interruption_context)
        } for replacement_state in
            list(itertools.product(*[
                [
                    {
                        'disruptor': disruptor,
                        'active': True
                    },
                    {
                        'disruptor': disruptor,
                        'active': False
                    }
                ]
                for disruptor in replacement_disruptors]))]

    # build service blacklists from replacement context
    for replacement_state in replacement_context:
        blacklisted_replacement_services = set()
        for disruptor_state in replacement_state['state']:
            if disruptor_state['active']:
                blacklisted_replacement_services.add(
                    next(filter(lambda service: disruptor_state['disruptor']['name'] in [disruptor['name'] for disruptor in service['disruptors']], g_services))['name'])

        if blacklisted_replacement_services in [blacklist_element['blacklisted_services'] for blacklist_element in replacement_blacklist]:
            next(filter(lambda blacklist_element: blacklist_element['blacklisted_services'] == blacklisted_replacement_services,
                        replacement_blacklist))['probability'] += replacement_state['probability']
        else:
            replacement_blacklist.append({
                'blacklisted_services': blacklisted_replacement_services,
                'probability': replacement_state['probability']
            })

    return sorted(replacement_blacklist, key=lambda rb: len(rb['blacklisted_services']))


# returns a list of interruption possibilities for interrupted_service (blacklist is only needed for performance improvements in creating the blacklist of immediate replacements)
def create_interruption_possibilities(interrupted_service, blacklist):
    p_intsingle = calculate_service_interruption_probability(interrupted_service)

    # if the services interruption probability is 0, there are no interruption possibilities present
    if p_intsingle == 0:
        return []

    replacement_blacklist = get_replacement_blacklist(interrupted_service, blacklist)

    interruption_possibilities = []
    # for each point during the duration of the interruptible service in which it can be interrupted (based on the exogenous resolution), add the state in which the service would be interrupted
    for interruption_time in range(0, g_service_data[interrupted_service['name']]['avg_duration'], g_exogenous_resolution):
        for replacement_blacklist_element in replacement_blacklist:
            occurrence_probability = p_intsingle * pow(1 - p_intsingle, interruption_time / g_exogenous_resolution)

            # for the interruption possibility at t=0, the interruption states must be considered because the immediate replacement service is subject to the same context as the replaced service
            soft_blacklist = replacement_blacklist_element['blacklisted_services']
            occurrence_probability *= replacement_blacklist_element['probability']

            # exclude interruption states whose probability is basically zero (floating point error)
            if occurrence_probability <= 1e-15:
                continue

            interruption_possibility = {
                "interrupted_service": deepcopy_dictionary(interrupted_service),
                "occurrence_probability": occurrence_probability,
                "interruption_state": deepcopy_dictionary(interrupted_service['state_before']),
                "soft_blacklist": soft_blacklist
            }

            # apply state & service modifications occurring during execution
            interruption_possibility['interrupted_service']['execution_percentage'] = interruption_time / g_service_data[interrupted_service['name']]['avg_duration']
            interruption_possibility['interruption_state']['time'] = interrupted_service['state_before']['time'] + interruption_time

            interruption_possibilities.append(deepcopy_dictionary(interruption_possibility))

    return interruption_possibilities


# returns all possible (that could happen in reality) service compositions based on the starting_state, hard_blacklist and soft_blacklist
# hard_blacklist: the list of services not to be considered for any further compositions in this composition tree (the first service in there is always the one whose interruption led to the part of the composition tree to be calculated)
# soft_blacklist: the list of services not to be considered for the immediately created compositions
def get_service_compositions(starting_state, hard_blacklist, soft_blacklist):
    service_compositions = []

    interrupted_service_class = g_service_data[hard_blacklist[0]]['service_class'] if hard_blacklist else None

    services_to_use = list(filter(lambda s: s['name'] not in [blacklisted_service_name for blacklisted_service_name in hard_blacklist + soft_blacklist] and
                                  (interrupted_service_class is None or s['service_class'] >= interrupted_service_class), g_services))

    # if there are no non-blacklisted services that may serve as an immediate replacement to an interrupted service, mark the composition as no further services (NFS)
    if interrupted_service_class is not None and len(list(filter(lambda s: s['service_class'] == interrupted_service_class, services_to_use))) == 0:
        return [
            {
                'status': "NFS",
                'services': [],
                'execution_probability': 1,
                'interruption_possibilities': []
            }
        ]

    for service_combination in itertools.product(*[list(sc_services) for sc, sc_services in itertools.groupby(services_to_use, lambda s: s['service_class'])]):
        service_composition = {
            'status': "Valid",
            'services': [{
                'name': service['name']
            } for service in service_combination],
            'execution_probability': 1,
            'interruption_possibilities': []
        }

        current_state = deepcopy_dictionary(starting_state)
        composition_is_valid = True

        services_up_until_now = []

        for service in service_composition['services']:
            # Composition validity checks
            if (current_state['time'] < g_service_data[service['name']]['business_hours']['opening'] or
                current_state['time'] + g_service_data[service['name']]['avg_duration'] > g_service_data[service['name']]['business_hours']['closing']):
                composition_is_valid = False
                break

            service['state_before'] = deepcopy_dictionary(current_state)
            service['execution_percentage'] = 1.0


            # for each possible interruption state, calculate the optimal alternative service composition
            for interruption_possibility in create_interruption_possibilities(service, hard_blacklist + soft_blacklist):
                interruption_possibility['alternative_composition_candidates'] = get_service_compositions(interruption_possibility['interruption_state'],
                                                                                                          [service['name']] + hard_blacklist,
                                                                                                          interruption_possibility['soft_blacklist'])

                # no alternative composition candidates could be found (e.g. due to context effects like business hours
                if len(interruption_possibility['alternative_composition_candidates']) == 0:
                    interruption_possibility['alternative_composition_candidates'] = [
                        {
                            'status': 'NAC',
                            'services': [],
                            'execution_probability': 1,
                            'interruption_possibilities': []
                        }
                    ]

                for alternative_service_composition_candidate in interruption_possibility['alternative_composition_candidates']:
                    alternative_service_composition_candidate['parent_interruption_possibility'] = interruption_possibility

                    for alternative_service_composition_candidate_subcomposition in get_self_and_subcomposition_candidates(alternative_service_composition_candidate):
                        alternative_service_composition_candidate_subcomposition['execution_probability'] *= interruption_possibility['occurrence_probability']
                        alternative_service_composition_candidate_subcomposition['services'] = services_up_until_now + [interruption_possibility['interrupted_service']] + alternative_service_composition_candidate_subcomposition['services']

                service_composition['execution_probability'] -= interruption_possibility['occurrence_probability']
                service_composition['interruption_possibilities'].append(interruption_possibility)

            # State modifications after complete service execution
            current_state['time'] += g_service_data[service['name']]['avg_duration']
            current_state['discounts'].extend(g_service_data[service['name']]['discounts'])
            for discount_to_apply in filter(lambda discount: discount['service'] == service['name'], current_state['discounts']):
                current_state['discounts'].remove(discount_to_apply)

            services_up_until_now.append(service)

        if composition_is_valid:
            service_compositions.append(service_composition)

    return service_compositions


# prints out the global compositions and their expected utilities
def print_all_compositions(limit):
    sorted_global_compositions = sorted(global_service_compositions, key=lambda gsc: gsc['expected_utility'], reverse=True)
    if limit > 0:
        sorted_global_compositions = sorted_global_compositions[:limit]

    longest_service_composition = max(sorted_global_compositions, key=lambda gsc: len(' - '.join([service['name'] for service in gsc['services']])))
    max_service_composition_string_length = len(' - '.join([service['name'] for service in longest_service_composition['services']]))

    service_padding = (20 - max_service_composition_string_length)
    service_padding = (math.ceil(service_padding / 2), math.floor(service_padding / 2)) if service_padding % 2 != 0 else (service_padding / 2, service_padding / 2)

    print('+{}+'.format('-' * (41 + max(max_service_composition_string_length - 20, 0))))
    print('| Composition services{} | Expected utility |'.format(' ' * (max_service_composition_string_length - 20)))
    for global_service_composition in sorted_global_compositions:
        print('| {}{}{} |     {:.6f}     |'.
              format(' ' * service_padding[0],
                     ' - '.join([service['name'] for service in global_service_composition['services']]),
                     ' ' * service_padding[1],
                     global_service_composition['expected_utility']))
    print('+{}+'.format('-' * (41 + max(max_service_composition_string_length - 20, 0))))


# prints a composition and its attributes with its alternative compositions shortened and indented to indicate interruptions, services denoted with 'i' (interrupted) were interrupted after execution start, service denoted with 'o' (omitted) before
def print_composition(service_composition):
    if 'parent_interruption_possibility' in service_composition:
        first_original_service_index = service_composition['services'].index(service_composition['parent_interruption_possibility']['interrupted_service'])
    else:
        first_original_service_index = 0

    services_to_print = [service for service in service_composition['services'][first_original_service_index:]]
    if service_composition['status'] in ['NFS', 'NAC']:
        services_to_print.append({
            'name': service_composition['status'],
            'execution_percentage': 1
        })

    indentation_level = service_composition['services'].index(services_to_print[0])

    print('{0}{1} [TEP: {3:6.2f}%]{2}[ExP: {4:6.2f}%, U: {5:.4f}, ExU: {6:.4f}]'
          .format('       ' * indentation_level,
                  ' - '.join([service['name'] + ('o' if service['execution_percentage'] == 0 else ('i' if service['execution_percentage'] < 1 else ' ')) for service in services_to_print]),
                  '       ' * (8 - len(services_to_print) - indentation_level),
                  sum(sc['execution_probability'] for sc in get_self_and_subcompositions(service_composition)) * 100,
                  abs(service_composition['execution_probability'] * 100),
                  service_composition['utility'],
                  service_composition['expected_utility']))

    for interruption_possibility in service_composition.get('interruption_possibilities', []):
        print_composition(interruption_possibility['alternative_composition'])


# load the problem data from file
def load_data(filename):
    with open(filename, 'r') as data_file:
        json_data = ujson.loads(data_file.read())

    for key in ['services', 'utility_factor_weights', 'user_favourites', 'xcvs', 'initial_state', 'exogenous_resolution']:
        if key not in json_data:
            print("Lacking data from configuration file, key \"" + key + "\" not found")

    global g_services, g_utility_factors, g_user_favourites, g_xcvs, g_initial_state, g_exogenous_resolution, g_service_data

    for utility_factor_name, utility_factor in g_utility_factors.items():
        utility_factor['weight'] = json_data['utility_factor_weights'][utility_factor_name]

    g_user_favourites = json_data['user_favourites']

    g_services = json_data['services']
    for service in g_services:
        service['type_score'] = g_user_favourites[service['type']]
        for disruptor in service['disruptors']:
            if isinstance(disruptor['conditions'], str) and disruptor['conditions'].startswith('interval'):
                disruptor['conditions'] = eval(disruptor['conditions'])
    g_service_data = {service['name']: service for service in g_services}

    g_xcvs = json_data['xcvs']

    g_initial_state = json_data['initial_state']

    g_exogenous_resolution = json_data['exogenous_resolution']


# global variable definitions
g_services = []
g_service_data = {}
g_user_favourites = {}
g_xcvs = {}
g_initial_state = {}
g_exogenous_resolution = 0
g_utility_factors = {
    "total_cost": {
        "type": "min",
        "aggregation": "sum"
    },
    "total_type_score": {
        "type": "max",
        "aggregation": "weighted_sum (duration)"
    },
    "average_execution_percentage": {
        "type": "max",
        "aggregation": "weighted_average (duration)"
    }
}


try:
    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Calulates the optimal service composition for the given problem considering possible service interruptions.")
    parser.add_argument('--data-file', '-d', default='problem-data.json', help="The path of the file in which the problem data can be found")
    parser.add_argument('--limit-compositions', '-l', type=int, action='store', default=0, help="Limit the composition output to the top LIMIT_COMPOSITIONS compositions")
    parser.add_argument('--print-detailed-compositions', '-p', action='store_true', help="Prints out the exact make-up of each configuration")
    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        parser.print_help()
        exit(1)

    load_data(args.data_file)

    # create the base service compositions (complete enumeration)
    global_service_compositions = get_service_compositions(g_initial_state, [], [])

    # calculate the composition aggregates for every single service composition and alternative service composition
    possible_service_compositions = get_possible_service_compositions(global_service_compositions)
    for possible_service_composition in possible_service_compositions:
        calculate_composition_aggregates(possible_service_composition)

    # extract the extreme values from the list of possible service compositions
    aggregates_extreme_values = get_aggregates_extreme_values(possible_service_compositions)

    # calculate the exepcted utility for every global service composition
    for global_service_composition in global_service_compositions:
        calculate_composition_expected_utility(global_service_composition, aggregates_extreme_values)

    if args.print_detailed_compositions:
        for global_service_composition in global_service_compositions:
            print_composition(global_service_composition)
            print()

    print_all_compositions(args.limit_compositions)

except Exception as e:
    print("Unknown error occured, possibly check your input data?")
    raise e
