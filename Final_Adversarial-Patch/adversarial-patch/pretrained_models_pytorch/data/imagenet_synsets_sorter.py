with open('imagenet_synsets.txt' , 'r') as f:
    wnid_class_name = [line for line in f.readlines()[:1001] if line and line.startswith('n')]
sorted_wnid_class_name = sorted(wnid_class_name, key=lambda line: line.split()[0])

with open('sorted_imagenet_synsets.txt', 'w') as f:
    for line in sorted_wnid_class_name:
        f.write(line) 