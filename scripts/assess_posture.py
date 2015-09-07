import json
from pause import assessment
import sys

if __name__ == '__main__':
    name = sys.argv[1]
    filename = '/home/buschbapti/Documents/pause/data_test/'+name+'_test.json'
    with open(filename) as data_file:
        data = json.load(data_file)
    reba = assessment.Assessment(25)
    score = reba.assess(data)
    print 'name : {}, score = {}'.format(name,score)