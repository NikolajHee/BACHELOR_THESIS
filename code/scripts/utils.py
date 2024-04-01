

import os




def save_parameters(save_path, dictionary):
    with open(os.path.join(save_path, 'parameters.txt'), 'w') as f:
        for (key, value) in dictionary.items():
            f.write(f"{key} : {value}")
            f.write("\n")



if __name__ =='__main__':
    save_parameters('', test='hej', test2='hej2', test3='hej4')

