import os

file1 = open("datasets\\pairs-val.txt", "r")
file2 = open("datasets\\pairs-val2.txt", "w")

root = "datasets\\val"
for line in file1.read().split("\n"):
    line_split = line.split(" ")
    if not os.path.exists(os.path.join(root, line_split[0])):
        continue
    elif not os.path.exists(os.path.join(root, line_split[1])):
        continue
    elif not os.path.exists(os.path.join(root, 'keypoints/' + line_split[0].split('/')[1].split('.')[0] + '.txt')):
        continue
    elif not os.path.exists(os.path.join(root, 'image/' + line_split[1].split('/')[1].split('.')[0] + '.jpg')):
        continue
    else:
        write_line = ' '.join(line_split[:2])
        file2.write(write_line + '\n')
print()