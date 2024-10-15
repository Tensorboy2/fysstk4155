'''
The main file of the project.
This file should run all nessesary code to produce all results.
'''
from data.prepare_data import prepare_data


def main():
    '''Main function'''
    x_train, x_test, y_train, y_test = prepare_data()

if __name__ == '__main__':
    main()
