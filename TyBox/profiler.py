
from create_l import create_layer


class Profiler:
    def __init__(self, network_name, model, mem_optimization=1, precisions=[[32, 32], [32, 32]]):
        self.network_name = network_name
        self.model = model
        self.mem_optimization = mem_optimization
        self.precisions = precisions

        self.per_layer_results = None
        self.tot_parameters = None
        self.peak_activations = None
        self.tot_parameters_memory = None
        self.peak_activations_memory = None

        self.evaluate_model()
        self.calculate_total_occupations()

        self.tot_mem_required = self.peak_activations_memory[0] \
                                + self.peak_activations_memory[1] \
                                + self.tot_parameters_memory[0] + self.tot_parameters_memory[1]
    def evaluate_model(self):
        model = self.model
        i = 0
        is_class_head = 0

        per_layer_results = []

        while True:
            try:
                layer = model.get_layer(index=i)
                print(type(layer))
            except Exception as e:
                print(e)
                print('no more layers (or error)')
                break
            print(f'evaluating {layer}')

            result = create_layer(layer)
            name = result.get_name()
            print(name)
            if name == 'Flatten' or name == 'Dense':
                is_class_head = 1
            mem_results = result.get_sizes()
            op_results = result.get_operations()

            per_layer_results.append([mem_results, op_results, name, is_class_head])
            i += 1

        self.per_layer_results = per_layer_results

    def calculate_total_occupations(self):
        tot_parameters = [0, 0]
        tot_parameters_memory = [0, 0]
        peak_activations = [0, 0]
        peak_activations_memory = [0, 0]

        per_layer_results = self.per_layer_results
        precisions = self.precisions

        for r in range(len(per_layer_results)):
            is_class_head = per_layer_results[r][3]
            tot_parameters[is_class_head] = per_layer_results[r][0]['param'] + tot_parameters[is_class_head]
            tot_parameters_memory[is_class_head] = per_layer_results[r][0]['param'] * precisions[is_class_head][0]/8 \
                                                   + tot_parameters_memory[is_class_head]

        if self.mem_optimization:
            peak_act_fe = 0
            peak_act_fe_mem = 0
            for r in range(len(per_layer_results)-1):
                if not per_layer_results[r+1][3]:
                    if (per_layer_results[r][0]['act'] + per_layer_results[r+1][0]['act']) > peak_act_fe:
                        peak_act_fe = per_layer_results[r][0]['act'] + per_layer_results[r+1][0]['act']
                        peak_act_fe_mem = (per_layer_results[r][0]['act'] + per_layer_results[r+1][0]['act']) \
                                          * precisions[0][1]/8
        else:
            peak_act_fe = 0
            peak_act_fe_mem = 0
            for r in range(len(per_layer_results)):
                if not per_layer_results[r][3]:
                    peak_act_fe = per_layer_results[r][0]['act'] + peak_act_fe
                    peak_act_fe_mem = per_layer_results[r][0]['act'] * precisions[0][1]/8 + peak_act_fe_mem

        peak_activations[0] = peak_act_fe
        peak_activations_memory[0] = peak_act_fe_mem


        peak_act_class = 0
        peak_act_class_mem = 0
        for r in range(len(per_layer_results)):
            if per_layer_results[r][3]:
                peak_act_class = per_layer_results[r][0]['act'] + peak_act_class
                peak_act_class_mem = per_layer_results[r][0]['act'] * precisions[1][1]/8 + peak_act_class_mem
        peak_activations[1] = peak_act_class
        peak_activations_memory[1] = peak_act_class_mem

        self.tot_parameters = tot_parameters
        self.peak_activations = peak_activations
        self.tot_parameters_memory = tot_parameters_memory
        self.peak_activations_memory = peak_activations_memory

    def print_occupations(self):

        print('total parameters fe: ', self.tot_parameters[0])
        print('total parameters classification head:', self.tot_parameters[1])
        print('total parameters: ', self.tot_parameters[0]+self.tot_parameters[1])
        print(f'peak activations fe: {self.peak_activations[0]}')
        print(f'peak activations classification head: {self.peak_activations[1]}')
        print(f'peak activations total: {self.peak_activations[0] + self.peak_activations[1]}')

        print(f'total parameters fe mem:  {self.tot_parameters_memory[0]} B')
        print(f'total parameters classification head mem: {self.tot_parameters_memory[1]} B')
        print(f'total parameters mem: {self.tot_parameters_memory[0]+self.tot_parameters_memory[1]} B')
        print(f'peak activations fe mem: {self.peak_activations_memory[0]} B')
        print(f'peak activations classification head mem: {self.peak_activations_memory[1]} B')
        print(f'peak activations total mem: {self.peak_activations_memory[0] + self.peak_activations_memory[1]} B')

    def print_per_layer(self):

        for l in self.per_layer_results:
            print(l[2])
            print(f"memory = {l[0]}")
            print(f"operations = {l[1]}")