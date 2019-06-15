import SegmentCharacters
import pickle
import matplotlib.pyplot as plt

filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
fig, ax1 = plt.subplots(1)

classification_result = []
for each_character in SegmentCharacters.characters:
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    classification_result.append(result)


plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

column_list_copy = SegmentCharacters.column_list[:]
SegmentCharacters.column_list.sort()
rightplate_string = ''
for each in SegmentCharacters.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print('Placa encontrada: ' + rightplate_string)